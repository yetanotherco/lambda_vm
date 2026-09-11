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
/// Multilinear evaluations bound on device.
static EVALUATE_CALLS: AtomicU64 = AtomicU64::new(0);
/// Fraction trees built and kept on device.
static TREE_CALLS: AtomicU64 = AtomicU64::new(0);
/// Tables whose factors were uploaded once and reused.
static FACTOR_CALLS: AtomicU64 = AtomicU64::new(0);
/// Openings whose two factors stayed on device across their groups.
static OPEN_CALLS: AtomicU64 = AtomicU64::new(0);

pub fn commit_calls() -> u64 {
    COMMIT_CALLS.load(Ordering::Relaxed)
}

pub fn sumcheck_calls() -> u64 {
    SUMCHECK_CALLS.load(Ordering::Relaxed)
}

pub fn sumcheck_rounds() -> u64 {
    SUMCHECK_ROUNDS.load(Ordering::Relaxed)
}

pub fn evaluate_calls() -> u64 {
    EVALUATE_CALLS.load(Ordering::Relaxed)
}

pub fn tree_calls() -> u64 {
    TREE_CALLS.load(Ordering::Relaxed)
}

pub fn factor_calls() -> u64 {
    FACTOR_CALLS.load(Ordering::Relaxed)
}

pub fn open_calls() -> u64 {
    OPEN_CALLS.load(Ordering::Relaxed)
}

pub fn reset_call_counters() {
    COMMIT_CALLS.store(0, Ordering::Relaxed);
    SUMCHECK_CALLS.store(0, Ordering::Relaxed);
    SUMCHECK_ROUNDS.store(0, Ordering::Relaxed);
    EVALUATE_CALLS.store(0, Ordering::Relaxed);
    TREE_CALLS.store(0, Ordering::Relaxed);
    FACTOR_CALLS.store(0, Ordering::Relaxed);
    OPEN_CALLS.store(0, Ordering::Relaxed);
}

/// A sumcheck's round proofs, the challenges they drew, and the factors the
/// rounds left bound.
type SumcheckRounds<E> = (
    Vec<crate::sumcheck::RoundProof<E>>,
    Vec<math::field::element::FieldElement<E>>,
    Vec<crate::mle::Mle<E>>,
);

/// Codeword size below which the host wins: the kernels are a dozen launches
/// and a round trip, and a small NTT finishes in host cache before that.
#[cfg(feature = "cuda")]
const COMMIT_THRESHOLD: usize = 1 << 16;

/// A byte buffer of Merkle nodes, relabelled as nodes without copying.
///
/// A tree over a stacked polynomial is hundreds of megabytes; chunking it into
/// arrays would copy all of it to change nothing but the type.
#[cfg(feature = "cuda")]
fn nodes_in_place(bytes: Vec<u8>) -> Option<Vec<[u8; 32]>> {
    if !bytes.len().is_multiple_of(32) || !bytes.capacity().is_multiple_of(32) {
        return None;
    }
    // SAFETY: `[u8; 32]` has the alignment of `u8` and 32 times its size, so
    // the allocation describes the same bytes either way.
    let mut bytes = core::mem::ManuallyDrop::new(bytes);
    Some(unsafe {
        Vec::from_raw_parts(
            bytes.as_mut_ptr() as *mut [u8; 32],
            bytes.len() / 32,
            bytes.capacity() / 32,
        )
    })
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
    resident: Option<&DeviceFactors>,
    program: &crate::program::Program<E>,
    degree: usize,
    challenge: impl FnMut(
        &[math::field::element::FieldElement<E>],
    ) -> math::field::element::FieldElement<E>,
) -> Option<Result<SumcheckRounds<E>, crate::Error>>
where
    E: math::field::traits::IsField + 'static,
{
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

    // Factors already on device are folded where they are; the rest — a
    // batch's weight tables — go up with them.
    let session = match resident {
        Some(factors) => {
            let held = factors.0.width();
            if held > raw.len() || factors.0.len() != first.len() {
                return None;
            }
            factors.0.session(
                &raw[held..],
                &lowered.nodes,
                &lowered.consts,
                lowered.num_slots,
                lowered.root_slot,
            )
        }
        None => math_cuda::sumcheck::SumcheckSession::new(
            &raw,
            &lowered.nodes,
            &lowered.consts,
            lowered.num_slots,
            lowered.root_slot,
        ),
    };
    let mut session = session.ok()?;

    // Diagnostic hook: recompute each round on the host from the factors the
    // device holds and stop at the first disagreement, naming the round. A
    // device round that differs otherwise surfaces as a proof that does not
    // verify, minutes and 55 tables later.
    static XCHECK: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    let xcheck = *XCHECK.get_or_init(|| std::env::var_os("LAMBDA_VM_GPU_XCHECK").is_some());
    let reference = |session: &math_cuda::sumcheck::SumcheckSession| {
        // A session over factors that were already there has no layout to read
        // back, so there is nothing to rebuild the host round from.
        if !xcheck || !session.can_download() {
            return None;
        }
        let tables = session.download().expect("the device holds its factors");
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
    };

    let outcome = run_rounds(&mut session, degree, num_vars, challenge, reference);
    let (rounds, challenges) = match outcome {
        Ok(rounds) => rounds,
        Err(error) => return Some(Err(error)),
    };
    // Every variable is bound, so each factor is one value — read where it
    // lies, which is the only thing a session over resident factors can say.
    let Ok(bound) = session.bound_values() else {
        return Some(Err(crate::Error::DeviceFailed { stage: "download" }));
    };
    let folded: Result<Vec<crate::mle::Mle<E>>, crate::Error> = bound
        .iter()
        .map(|value| crate::mle::Mle::new(vec![ext3_from_raw::<E>(value)]))
        .collect();
    let Ok(folded) = folded else {
        return Some(Err(crate::Error::DeviceFailed {
            stage: "folded tables",
        }));
    };
    SUMCHECK_CALLS.fetch_add(1, Ordering::Relaxed);
    Some(Ok((rounds, challenges, folded)))
}

/// The round loop: the device sums over the cube, the host draws the challenge
/// from what it sent, the device binds it.
///
/// Past the first round the transcript has moved, so every failure here is an
/// error — there is no going back to the host path.
#[cfg(feature = "cuda")]
fn run_rounds<E>(
    session: &mut math_cuda::sumcheck::SumcheckSession,
    degree: usize,
    num_vars: usize,
    mut challenge: impl FnMut(
        &[math::field::element::FieldElement<E>],
    ) -> math::field::element::FieldElement<E>,
    mut reference: impl FnMut(
        &math_cuda::sumcheck::SumcheckSession,
    ) -> Option<Vec<math::field::element::FieldElement<E>>>,
) -> Result<
    (
        Vec<crate::sumcheck::RoundProof<E>>,
        Vec<math::field::element::FieldElement<E>>,
    ),
    crate::Error,
>
where
    E: math::field::traits::IsField + 'static,
{
    use math::field::element::FieldElement;

    // The interpolation nodes are `1..=degree`: `g(0)` is not sent, the claim
    // carried into the round fixes it.
    let mut t = Vec::with_capacity(degree * 3);
    for node in 1..=degree {
        t.extend_from_slice(&ext3_raw(&FieldElement::<E>::from(node as u64)).ok_or(
            crate::Error::DeviceFailed {
                stage: "interpolation node",
            },
        )?);
    }

    let failed = |stage| crate::Error::DeviceFailed { stage };
    let mut rounds = Vec::with_capacity(num_vars);
    let mut challenges = Vec::with_capacity(num_vars);
    for round in 0..num_vars {
        let expected = reference(session);
        let sums = session.round(&t).map_err(|_| failed("round"))?;
        let evaluations: Vec<FieldElement<E>> =
            sums.chunks_exact(3).map(ext3_from_raw::<E>).collect();
        if let Some(expected) = expected {
            assert_eq!(
                evaluations, expected,
                "the device round {round} differs from the host"
            );
        }
        let r = challenge(&evaluations);
        let raw = ext3_raw(&r).ok_or_else(|| failed("challenge"))?;
        session.fold(&raw).map_err(|_| failed("fold"))?;
        rounds.push(crate::sumcheck::RoundProof { evaluations });
        challenges.push(r);
        SUMCHECK_ROUNDS.fetch_add(1, Ordering::Relaxed);
    }
    Ok((rounds, challenges))
}

#[cfg(not(feature = "cuda"))]
pub(crate) fn prove_sumcheck<E>(
    _polys: &[crate::mle::Mle<E>],
    _resident: Option<&DeviceFactors>,
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
    let nodes = nodes_in_place(nodes)?;
    COMMIT_CALLS.fetch_add(1, Ordering::Relaxed);
    Some(nodes)
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

/// Table size below which the host evaluation wins: one launch per variable
/// against a couple of passes a few cores finish in microseconds.
#[cfg(feature = "cuda")]
const EVALUATE_THRESHOLD: usize = 1 << 16;

/// A multilinear's value at `point`, bound variable by variable on device.
///
/// `evals` may be base-field or ext3; the point is always ext3, which is what
/// a challenge is.
#[cfg(feature = "cuda")]
pub(crate) fn evaluate_mle<C, E>(
    evals: &[math::field::element::FieldElement<C>],
    point: &[math::field::element::FieldElement<E>],
) -> Option<math::field::element::FieldElement<E>>
where
    C: math::field::traits::IsField + 'static,
    E: math::field::traits::IsField + 'static,
{
    use math::field::extensions_goldilocks::Degree3GoldilocksExtensionField as Ext3;
    use math::field::goldilocks::GoldilocksField as Gl;
    use std::any::TypeId;

    if TypeId::of::<E>() != TypeId::of::<Ext3>() {
        return None;
    }
    let base = TypeId::of::<C>() == TypeId::of::<Gl>();
    if !base && TypeId::of::<C>() != TypeId::of::<Ext3>() {
        return None;
    }
    if point.is_empty() || evals.len() < EVALUATE_THRESHOLD || evals.len() != 1 << point.len() {
        return None;
    }
    static DISABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    if *DISABLED.get_or_init(|| std::env::var_os("LAMBDA_VM_NO_GPU_MLE_EVAL").is_some()) {
        return None;
    }

    let mut raw_point = Vec::with_capacity(point.len() * 3);
    for coordinate in point {
        raw_point.extend_from_slice(&ext3_raw(coordinate)?);
    }
    // SAFETY: `C` is one of the two fields checked above, and both wrap their
    // limbs transparently — one `u64` per base element, three per ext3.
    let limbs = if base { 1 } else { 3 };
    let raw =
        unsafe { core::slice::from_raw_parts(evals.as_ptr() as *const u64, evals.len() * limbs) };
    let value = if base {
        math_cuda::sumcheck::evaluate_mle_base(raw, &raw_point).ok()?
    } else {
        math_cuda::sumcheck::evaluate_mle_ext3(raw, &raw_point).ok()?
    };
    EVALUATE_CALLS.fetch_add(1, Ordering::Relaxed);
    Some(ext3_from_raw::<E>(&value))
}

#[cfg(not(feature = "cuda"))]
pub(crate) fn evaluate_mle<C, E>(
    _evals: &[math::field::element::FieldElement<C>],
    _point: &[math::field::element::FieldElement<E>],
) -> Option<math::field::element::FieldElement<E>>
where
    C: math::field::traits::IsField + 'static,
    E: math::field::traits::IsField + 'static,
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

/// Input-layer size below which the host tree wins: the levels are a launch
/// each and the fold is a pass a few cores finish in microseconds.
#[cfg(feature = "cuda")]
const TREE_THRESHOLD: usize = 1 << 14;

/// A LogUp fraction tree the device holds, layers and all.
#[cfg(feature = "cuda")]
pub struct DeviceTree(math_cuda::gkr::DeviceFractionTree);

#[cfg(feature = "cuda")]
impl std::fmt::Debug for DeviceTree {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DeviceTree")
            .field("layers", &self.0.num_layers())
            .finish()
    }
}

/// A device tree the build declined to make. Never constructed.
#[cfg(not(feature = "cuda"))]
#[derive(Debug)]
pub struct DeviceTree(std::convert::Infallible);

#[cfg(not(feature = "cuda"))]
impl DeviceTree {
    pub(crate) fn num_layers(&self) -> usize {
        match self.0 {}
    }

    pub(crate) fn output<E>(
        &self,
    ) -> Result<
        (
            math::field::element::FieldElement<E>,
            math::field::element::FieldElement<E>,
        ),
        crate::Error,
    >
    where
        E: math::field::traits::IsField + 'static,
    {
        match self.0 {}
    }

    pub(crate) fn prove_layer<E>(
        &self,
        _layer: usize,
        _point: &[math::field::element::FieldElement<E>],
        _program: &crate::program::Program<E>,
        _degree: usize,
        _challenge: impl FnMut(
            &[math::field::element::FieldElement<E>],
        ) -> math::field::element::FieldElement<E>,
    ) -> Option<Result<LayerRounds<E>, crate::Error>>
    where
        E: math::field::traits::IsField + 'static,
    {
        match self.0 {}
    }
}

/// What a layer's sumcheck leaves: the rounds, the point, and the four values
/// the layer reduces to.
pub type LayerRounds<E> = (
    Vec<crate::sumcheck::RoundProof<E>>,
    Vec<math::field::element::FieldElement<E>>,
    [math::field::element::FieldElement<E>; 4],
);

/// Builds the tree on device from an input layer, folding every level there.
#[cfg(feature = "cuda")]
pub(crate) fn build_tree<E>(p: &crate::mle::Mle<E>, q: &crate::mle::Mle<E>) -> Option<DeviceTree>
where
    E: math::field::traits::IsField + 'static,
{
    use math::field::extensions_goldilocks::Degree3GoldilocksExtensionField as Ext3;

    if std::any::TypeId::of::<E>() != std::any::TypeId::of::<Ext3>() {
        return None;
    }
    if p.len() < TREE_THRESHOLD || p.len() != q.len() {
        return None;
    }
    static DISABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    if *DISABLED.get_or_init(|| std::env::var_os("LAMBDA_VM_NO_GPU_GKR").is_some()) {
        return None;
    }
    // SAFETY: `E == Ext3`, three transparent `u64` limbs per element.
    let raw = |table: &crate::mle::Mle<E>| unsafe {
        core::slice::from_raw_parts(table.evals().as_ptr() as *const u64, table.len() * 3)
    };
    let tree = math_cuda::gkr::DeviceFractionTree::build(raw(p), raw(q)).ok()?;
    TREE_CALLS.fetch_add(1, Ordering::Relaxed);
    Some(DeviceTree(tree))
}

#[cfg(not(feature = "cuda"))]
pub(crate) fn build_tree<E>(_p: &crate::mle::Mle<E>, _q: &crate::mle::Mle<E>) -> Option<DeviceTree>
where
    E: math::field::traits::IsField + 'static,
{
    None
}

#[cfg(feature = "cuda")]
impl DeviceTree {
    pub(crate) fn num_layers(&self) -> usize {
        self.0.num_layers()
    }

    /// The output fraction, which is what says whether the bus balances.
    pub(crate) fn output<E>(
        &self,
    ) -> Result<
        (
            math::field::element::FieldElement<E>,
            math::field::element::FieldElement<E>,
        ),
        crate::Error,
    >
    where
        E: math::field::traits::IsField + 'static,
    {
        let (p, q) = self.0.output().map_err(|_| crate::Error::DeviceFailed {
            stage: "tree output",
        })?;
        Ok((ext3_from_raw::<E>(&p), ext3_from_raw::<E>(&q)))
    }

    /// One layer's sumcheck, folded in place: the rounds, the point they drew,
    /// and the four values the fold leaves behind.
    ///
    /// The layer is spent afterwards, which is what makes the halves usable as
    /// factors without copying them: GKR reads each layer once.
    pub(crate) fn prove_layer<E>(
        &self,
        layer: usize,
        point: &[math::field::element::FieldElement<E>],
        program: &crate::program::Program<E>,
        degree: usize,
        challenge: impl FnMut(
            &[math::field::element::FieldElement<E>],
        ) -> math::field::element::FieldElement<E>,
    ) -> Option<Result<LayerRounds<E>, crate::Error>>
    where
        E: math::field::traits::IsField + 'static,
    {
        let lowered = lower(program)?;
        let mut raw_point = Vec::with_capacity(point.len() * 3);
        for coordinate in point {
            raw_point.extend_from_slice(&ext3_raw(coordinate)?);
        }
        let num_vars = self.0.layer_num_vars(layer).checked_sub(1)?;
        let session = self.0.layer_sumcheck(
            layer,
            &raw_point,
            &lowered.nodes,
            &lowered.consts,
            lowered.num_slots,
            lowered.root_slot,
        );
        let Ok(mut session) = session else {
            return None;
        };

        // Past here the transcript moves: the host path is no longer an option.
        let failed = |stage| crate::Error::DeviceFailed { stage };
        let outcome = run_rounds(&mut session, degree, num_vars, challenge, |_| None);
        let (rounds, challenges) = match outcome {
            Ok(rounds) => rounds,
            Err(error) => return Some(Err(error)),
        };
        let Ok(bound) = session.bound_values() else {
            return Some(Err(failed("layer values")));
        };
        // Factor 0 is the weight; the four the layer reduces to follow.
        if bound.len() != 5 {
            return Some(Err(failed("layer factors")));
        }
        let value = |k: usize| ext3_from_raw::<E>(&bound[k]);
        SUMCHECK_CALLS.fetch_add(1, Ordering::Relaxed);
        Some(Ok((
            rounds,
            challenges,
            [value(1), value(2), value(3), value(4)],
        )))
    }
}

/// A table's factors, uploaded once for everything that walks them.
#[cfg(feature = "cuda")]
pub struct DeviceFactors(math_cuda::sumcheck::DeviceFactors);

#[cfg(feature = "cuda")]
impl std::fmt::Debug for DeviceFactors {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DeviceFactors")
            .field("factors", &self.0.width())
            .field("cells", &self.0.len())
            .finish()
    }
}

/// Factors a build declined to upload. Never constructed.
#[cfg(not(feature = "cuda"))]
#[derive(Debug)]
pub struct DeviceFactors(std::convert::Infallible);

/// Uploads a table's factors, or declines.
#[cfg(feature = "cuda")]
pub fn upload_factors<E>(factors: &[crate::mle::Mle<E>]) -> Option<DeviceFactors>
where
    E: math::field::traits::IsField + 'static,
{
    use math::field::extensions_goldilocks::Degree3GoldilocksExtensionField as Ext3;

    if std::any::TypeId::of::<E>() != std::any::TypeId::of::<Ext3>() {
        return None;
    }
    let first = factors.first()?;
    if first.len() < SUMCHECK_THRESHOLD || factors.iter().any(|f| f.len() != first.len()) {
        return None;
    }
    static DISABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    if *DISABLED.get_or_init(|| std::env::var_os("LAMBDA_VM_NO_GPU_FACTORS").is_some()) {
        return None;
    }
    // SAFETY: `E == Ext3`, three transparent `u64` limbs per element.
    let raw: Vec<&[u64]> = factors
        .iter()
        .map(|f| unsafe {
            core::slice::from_raw_parts(f.evals().as_ptr() as *const u64, f.len() * 3)
        })
        .collect();
    let uploaded = math_cuda::sumcheck::DeviceFactors::upload(&raw).ok()?;
    FACTOR_CALLS.fetch_add(1, Ordering::Relaxed);
    Some(DeviceFactors(uploaded))
}

#[cfg(not(feature = "cuda"))]
pub fn upload_factors<E>(_factors: &[crate::mle::Mle<E>]) -> Option<DeviceFactors>
where
    E: math::field::traits::IsField + 'static,
{
    None
}

/// The fraction tree's input layer, written by the device from the factors it
/// already holds, and folded into a tree without ever coming back.
///
/// Each interaction contributes a `p` and a `q` slab of `rows`, in the order
/// [`logup::input_layer`](crate::logup::input_layer) lays them out, with the
/// padding slots carrying `0/1`.
#[cfg(feature = "cuda")]
pub fn input_layer_tree<E>(
    factors: &DeviceFactors,
    numerators: &[crate::program::Program<E>],
    denominators: &[crate::program::Program<E>],
) -> Option<DeviceTree>
where
    E: math::field::traits::IsField + 'static,
{
    use math::field::element::FieldElement;

    let rows = factors.0.len();
    let slots = numerators.len().next_power_of_two();
    let stream = factors.0.stream().clone();
    let mut p = stream.alloc_zeros::<u64>(slots * rows * 3).ok()?;
    let mut q = stream.alloc_zeros::<u64>(slots * rows * 3).ok()?;

    for (i, (numerator, denominator)) in numerators.iter().zip(denominators).enumerate() {
        for (program, out) in [(numerator, &mut p), (denominator, &mut q)] {
            let lowered = lower(program)?;
            factors
                .0
                .map_program(
                    &lowered.nodes,
                    &lowered.consts,
                    lowered.num_slots,
                    lowered.root_slot,
                    out,
                    i * rows,
                )
                .ok()?;
        }
    }
    // The padding interactions: numerator zero (already), denominator one.
    let one = ext3_raw(&FieldElement::<E>::one())?;
    let padding = (slots - numerators.len()) * rows;
    math_cuda::sumcheck::fill_ext3(&stream, &mut q, numerators.len() * rows, padding, &one).ok()?;

    let tree = math_cuda::gkr::DeviceFractionTree::from_device(stream, p, q).ok()?;
    TREE_CALLS.fetch_add(1, Ordering::Relaxed);
    Some(DeviceTree(tree))
}

#[cfg(not(feature = "cuda"))]
pub fn input_layer_tree<E>(
    _factors: &DeviceFactors,
    _numerators: &[crate::program::Program<E>],
    _denominators: &[crate::program::Program<E>],
) -> Option<DeviceTree>
where
    E: math::field::traits::IsField + 'static,
{
    None
}

/// The opening's two factors on a device: the weight the chain carries and the
/// message it is opening, resident across groups of rounds.
#[cfg(feature = "cuda")]
pub struct OpeningFactors {
    session: math_cuda::whir_open::OpeningSession,
    lowered: Lowered,
}

/// Factors a build declined to upload. Never constructed.
#[cfg(not(feature = "cuda"))]
pub struct OpeningFactors(std::convert::Infallible);

/// Puts the chain's two factors on a device, lifting the message on the way.
///
/// `program` is the rule the rounds evaluate — the product of the two — which
/// the host path runs as a closure.
#[cfg(feature = "cuda")]
pub(crate) fn open_on_device<F, E>(
    message: &crate::mle::Mle<F>,
    weight: &crate::mle::Mle<E>,
    program: &crate::program::Program<E>,
) -> Option<OpeningFactors>
where
    F: math::field::traits::IsField + 'static,
    E: math::field::traits::IsField + 'static,
{
    use math::field::extensions_goldilocks::Degree3GoldilocksExtensionField as Ext3;
    use math::field::goldilocks::GoldilocksField as Gl;
    use std::any::TypeId;

    if TypeId::of::<F>() != TypeId::of::<Gl>() || TypeId::of::<E>() != TypeId::of::<Ext3>() {
        return None;
    }
    if message.len() != weight.len() || message.len() < SUMCHECK_THRESHOLD {
        return None;
    }
    static DISABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    if *DISABLED.get_or_init(|| std::env::var_os("LAMBDA_VM_NO_GPU_OPEN").is_some()) {
        return None;
    }
    let lowered = lower(program)?;

    // SAFETY: the fields are the two checked above, each wrapping its limbs
    // transparently — one `u64` per base element, three per ext3.
    let raw_weight = unsafe {
        core::slice::from_raw_parts(weight.evals().as_ptr() as *const u64, weight.len() * 3)
    };
    let raw_message = unsafe {
        core::slice::from_raw_parts(message.evals().as_ptr() as *const u64, message.len())
    };
    let session = math_cuda::whir_open::OpeningSession::new(raw_weight, raw_message).ok()?;
    OPEN_CALLS.fetch_add(1, Ordering::Relaxed);
    Some(OpeningFactors { session, lowered })
}

#[cfg(not(feature = "cuda"))]
pub(crate) fn open_on_device<F, E>(
    _message: &crate::mle::Mle<F>,
    _weight: &crate::mle::Mle<E>,
    _program: &crate::program::Program<E>,
) -> Option<OpeningFactors>
where
    F: math::field::traits::IsField + 'static,
    E: math::field::traits::IsField + 'static,
{
    None
}

#[cfg(feature = "cuda")]
impl OpeningFactors {
    pub(crate) fn num_vars(&self) -> usize {
        self.session.num_vars()
    }

    /// One group of rounds, with the host drawing each challenge.
    ///
    /// The factors are folded in place and stay for the next group, which is
    /// the whole point: they are the width of a stacked polynomial.
    pub(crate) fn rounds<E>(
        &mut self,
        group: usize,
        degree: usize,
        challenge: impl FnMut(
            &[math::field::element::FieldElement<E>],
        ) -> math::field::element::FieldElement<E>,
    ) -> Result<SumcheckRounds<E>, crate::Error>
    where
        E: math::field::traits::IsField + 'static,
    {
        let failed = |stage| crate::Error::DeviceFailed { stage };
        let mut session = self
            .session
            .sumcheck(
                &self.lowered.nodes,
                &self.lowered.consts,
                self.lowered.num_slots,
                self.lowered.root_slot,
            )
            .map_err(|_| failed("opening session"))?;
        let (rounds, challenges) = run_rounds(&mut session, degree, group, challenge, |_| None)?;
        self.session.bound(group);
        SUMCHECK_CALLS.fetch_add(1, Ordering::Relaxed);
        // The factors stay on device; the caller reads them through this
        // handle, not through the tables it no longer has.
        Ok((rounds, challenges, Vec::new()))
    }

    /// The message's value at `point`, which the out-of-domain answer needs.
    pub(crate) fn evaluate_message<E>(
        &self,
        point: &[math::field::element::FieldElement<E>],
    ) -> Result<math::field::element::FieldElement<E>, crate::Error>
    where
        E: math::field::traits::IsField + 'static,
    {
        let raw = raw_point(point).ok_or(crate::Error::DeviceFailed { stage: "ood point" })?;
        let value = self
            .session
            .evaluate_message(&raw)
            .map_err(|_| crate::Error::DeviceFailed { stage: "ood value" })?;
        Ok(ext3_from_raw::<E>(&value))
    }

    /// `weight += gamma · eq(point, ·)`: the weight the next group carries.
    pub(crate) fn add_scaled_eq<E>(
        &mut self,
        point: &[math::field::element::FieldElement<E>],
        gamma: &math::field::element::FieldElement<E>,
    ) -> Result<(), crate::Error>
    where
        E: math::field::traits::IsField + 'static,
    {
        let failed = |stage| crate::Error::DeviceFailed { stage };
        let raw = raw_point(point).ok_or_else(|| failed("weight point"))?;
        let scale = ext3_raw(gamma).ok_or_else(|| failed("weight scale"))?;
        self.session
            .add_scaled_eq(&raw, &scale)
            .map_err(|_| failed("weight"))
    }
}

/// A point as the limbs the kernels read.
#[cfg(feature = "cuda")]
fn raw_point<E>(point: &[math::field::element::FieldElement<E>]) -> Option<Vec<u64>>
where
    E: math::field::traits::IsField + 'static,
{
    let mut raw = Vec::with_capacity(point.len() * 3);
    for coordinate in point {
        raw.extend_from_slice(&ext3_raw(coordinate)?);
    }
    Some(raw)
}

#[cfg(not(feature = "cuda"))]
impl OpeningFactors {
    pub(crate) fn num_vars(&self) -> usize {
        match self.0 {}
    }

    pub(crate) fn rounds<E>(
        &mut self,
        _group: usize,
        _degree: usize,
        _challenge: impl FnMut(
            &[math::field::element::FieldElement<E>],
        ) -> math::field::element::FieldElement<E>,
    ) -> Result<SumcheckRounds<E>, crate::Error>
    where
        E: math::field::traits::IsField + 'static,
    {
        match self.0 {}
    }

    pub(crate) fn evaluate_message<E>(
        &self,
        _point: &[math::field::element::FieldElement<E>],
    ) -> Result<math::field::element::FieldElement<E>, crate::Error>
    where
        E: math::field::traits::IsField + 'static,
    {
        match self.0 {}
    }

    pub(crate) fn add_scaled_eq<E>(
        &mut self,
        _point: &[math::field::element::FieldElement<E>],
        _gamma: &math::field::element::FieldElement<E>,
    ) -> Result<(), crate::Error>
    where
        E: math::field::traits::IsField + 'static,
    {
        match self.0 {}
    }
}

/// The same with the weight written on device from its shares: a stacked
/// polynomial's weight is as wide as the polynomial, and sending it would cost
/// more than the rounds that read it.
#[cfg(feature = "cuda")]
pub(crate) fn open_shared<F, E>(
    message: &crate::mle::Mle<F>,
    shares: &[crate::stacked_eval::WeightShare<'_, E>],
    n_stack: usize,
    program: &crate::program::Program<E>,
) -> Option<OpeningFactors>
where
    F: math::field::traits::IsField + 'static,
    E: math::field::traits::IsField + 'static,
{
    use math::field::extensions_goldilocks::Degree3GoldilocksExtensionField as Ext3;
    use math::field::goldilocks::GoldilocksField as Gl;
    use std::any::TypeId;

    if TypeId::of::<F>() != TypeId::of::<Gl>() || TypeId::of::<E>() != TypeId::of::<Ext3>() {
        return None;
    }
    if message.num_vars() != n_stack || message.len() < SUMCHECK_THRESHOLD {
        return None;
    }
    static DISABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    if *DISABLED.get_or_init(|| std::env::var_os("LAMBDA_VM_NO_GPU_OPEN").is_some()) {
        return None;
    }
    let lowered = lower(program)?;
    let shares: Vec<(usize, Vec<u64>, [u64; 3])> = shares
        .iter()
        .map(|share| {
            Some((
                share.offset,
                raw_point(share.point)?,
                ext3_raw(&share.scale)?,
            ))
        })
        .collect::<Option<_>>()?;

    // SAFETY: `F == GoldilocksField`, a transparent wrapper over `u64`.
    let raw_message = unsafe {
        core::slice::from_raw_parts(message.evals().as_ptr() as *const u64, message.len())
    };
    let session =
        math_cuda::whir_open::OpeningSession::from_shares(&shares, message.len(), raw_message)
            .ok()?;
    OPEN_CALLS.fetch_add(1, Ordering::Relaxed);
    Some(OpeningFactors { session, lowered })
}

#[cfg(not(feature = "cuda"))]
pub(crate) fn open_shared<F, E>(
    _message: &crate::mle::Mle<F>,
    _shares: &[crate::stacked_eval::WeightShare<'_, E>],
    _n_stack: usize,
    _program: &crate::program::Program<E>,
) -> Option<OpeningFactors>
where
    F: math::field::traits::IsField + 'static,
    E: math::field::traits::IsField + 'static,
{
    None
}

/// A codeword the device holds: the commit leaves one there and the chain
/// folds it there, so the array itself never crosses the bus.
#[cfg(feature = "cuda")]
pub struct DeviceCodeword(math_cuda::whir::DeviceCodeword);

/// A codeword a commit declined to keep. Never constructed.
#[cfg(not(feature = "cuda"))]
pub struct DeviceCodeword(std::convert::Infallible);

#[cfg(feature = "cuda")]
impl std::fmt::Debug for DeviceCodeword {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DeviceCodeword")
            .field("elements", &self.0.elements())
            .field("base", &self.0.is_base())
            .finish()
    }
}

#[cfg(not(feature = "cuda"))]
impl std::fmt::Debug for DeviceCodeword {
    fn fmt(&self, _: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self.0 {}
    }
}

/// Commits a stacked polynomial with the codeword left on device.
#[cfg(feature = "cuda")]
pub(crate) fn commit_resident<F>(
    evals: &[math::field::element::FieldElement<F>],
    log_blowup: usize,
    log_folding: usize,
) -> Option<(DeviceCodeword, Vec<[u8; 32]>)>
where
    F: math::field::traits::IsField + 'static,
{
    use math::field::goldilocks::GoldilocksField;

    if std::any::TypeId::of::<F>() != std::any::TypeId::of::<GoldilocksField>() {
        return None;
    }
    if !evals.len().is_power_of_two() || evals.len() << log_blowup < COMMIT_THRESHOLD {
        return None;
    }
    static DISABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    if *DISABLED.get_or_init(|| std::env::var_os("LAMBDA_VM_NO_GPU_WHIR_COMMIT").is_some()) {
        return None;
    }
    // SAFETY: `F == GoldilocksField`, a transparent wrapper over `u64`.
    let raw = unsafe { core::slice::from_raw_parts(evals.as_ptr() as *const u64, evals.len()) };
    let (codeword, nodes) = math_cuda::whir::commit_codeword(raw, log_blowup, log_folding).ok()?;
    let nodes = nodes_in_place(nodes)?;
    COMMIT_CALLS.fetch_add(1, Ordering::Relaxed);
    Some((DeviceCodeword(codeword), nodes))
}

#[cfg(not(feature = "cuda"))]
pub(crate) fn commit_resident<F>(
    _evals: &[math::field::element::FieldElement<F>],
    _log_blowup: usize,
    _log_folding: usize,
) -> Option<(DeviceCodeword, Vec<[u8; 32]>)>
where
    F: math::field::traits::IsField + 'static,
{
    None
}

#[cfg(feature = "cuda")]
impl DeviceCodeword {
    pub(crate) fn elements(&self) -> usize {
        self.0.elements()
    }

    /// Folds it `alphas.len()` times, leaving the result on device too.
    pub(crate) fn fold<F, N>(
        &self,
        generator: &math::field::element::FieldElement<F>,
        alphas: &[math::field::element::FieldElement<N>],
    ) -> Option<Self>
    where
        F: math::field::traits::IsField + 'static,
        N: math::field::traits::IsField + 'static,
    {
        let (two_inv, g_invs, raw_alphas) = fold_scalars(generator, alphas)?;
        let folded = math_cuda::whir::fold_resident(&self.0, two_inv, &g_invs, &raw_alphas).ok()?;
        Some(Self(folded))
    }

    /// The tree over its fold blocks, in the host node layout.
    pub(crate) fn commit(&self, log_folding: usize) -> Option<Vec<[u8; 32]>> {
        let nodes = math_cuda::whir::commit_resident_ext3(&self.0, log_folding).ok()?;
        let nodes = nodes_in_place(nodes)?;
        COMMIT_CALLS.fetch_add(1, Ordering::Relaxed);
        Some(nodes)
    }

    /// The blocks `indices` open, gathered where they lie — one launch and one
    /// copy back for the whole round.
    pub(crate) fn cosets<F>(
        &self,
        indices: &[usize],
        num_leaves: usize,
        block: usize,
    ) -> Option<Vec<Vec<math::field::element::FieldElement<F>>>>
    where
        F: math::field::traits::IsField + 'static,
    {
        let indices: Vec<u64> = indices.iter().map(|index| *index as u64).collect();
        let raw = self.0.cosets(&indices, num_leaves, block).ok()?;
        let values = values_from_raw::<F>(&raw, self.0.is_base())?;
        Some(values.chunks_exact(block).map(<[_]>::to_vec).collect())
    }

    /// Its first value, which is what the last fold leaves behind.
    pub(crate) fn first<F>(&self) -> Option<math::field::element::FieldElement<F>>
    where
        F: math::field::traits::IsField + 'static,
    {
        let raw = self.0.first().ok()?;
        values_from_raw(&raw, false)?.into_iter().next()
    }
}

/// `two_inv`, each level's inverse generator, and the challenges as limbs.
#[cfg(feature = "cuda")]
fn fold_scalars<F, N>(
    generator: &math::field::element::FieldElement<F>,
    alphas: &[math::field::element::FieldElement<N>],
) -> Option<(u64, Vec<u64>, Vec<u64>)>
where
    F: math::field::traits::IsField + 'static,
    N: math::field::traits::IsField + 'static,
{
    use math::field::element::FieldElement;
    use math::field::extensions_goldilocks::Degree3GoldilocksExtensionField as Ext3;
    use math::field::goldilocks::GoldilocksField as Gl;
    use std::any::TypeId;

    if TypeId::of::<F>() != TypeId::of::<Gl>() || TypeId::of::<N>() != TypeId::of::<Ext3>() {
        return None;
    }
    if alphas.is_empty() {
        return None;
    }
    // SAFETY: `F == GoldilocksField`, a transparent wrapper over `u64`.
    let generator = unsafe { *(generator as *const _ as *const u64) };
    let generator = FieldElement::<Gl>::from_raw(generator);
    let two_inv = *FieldElement::<Gl>::from(2u64).inv().ok()?.value();
    let mut g_inv = generator.inv().ok()?;
    let mut g_invs = Vec::with_capacity(alphas.len());
    for _ in 0..alphas.len() {
        g_invs.push(*g_inv.value());
        g_inv = g_inv.square();
    }
    let mut raw = Vec::with_capacity(alphas.len() * 3);
    for alpha in alphas {
        raw.extend_from_slice(&ext3_raw(alpha)?);
    }
    Some((two_inv, g_invs, raw))
}

/// Limbs as field elements, one `u64` each for a base codeword and three for
/// an extension one.
#[cfg(feature = "cuda")]
fn values_from_raw<F>(raw: &[u64], base: bool) -> Option<Vec<math::field::element::FieldElement<F>>>
where
    F: math::field::traits::IsField + 'static,
{
    use math::field::element::FieldElement;
    use math::field::extensions_goldilocks::Degree3GoldilocksExtensionField as Ext3;
    use math::field::goldilocks::GoldilocksField as Gl;
    use std::any::TypeId;

    if base {
        if TypeId::of::<F>() != TypeId::of::<Gl>() {
            return None;
        }
        return Some(
            raw.iter()
                .map(|limb| {
                    let value = FieldElement::<Gl>::from_raw(*limb);
                    // SAFETY: `F == Gl`, checked above; same representation.
                    unsafe {
                        core::mem::transmute_copy::<FieldElement<Gl>, FieldElement<F>>(&value)
                    }
                })
                .collect(),
        );
    }
    if TypeId::of::<F>() != TypeId::of::<Ext3>() {
        return None;
    }
    Some(raw.chunks_exact(3).map(ext3_from_raw::<F>).collect())
}

#[cfg(not(feature = "cuda"))]
impl DeviceCodeword {
    pub(crate) fn elements(&self) -> usize {
        match self.0 {}
    }

    pub(crate) fn fold<F, N>(
        &self,
        _generator: &math::field::element::FieldElement<F>,
        _alphas: &[math::field::element::FieldElement<N>],
    ) -> Option<Self>
    where
        F: math::field::traits::IsField + 'static,
        N: math::field::traits::IsField + 'static,
    {
        match self.0 {}
    }

    pub(crate) fn commit(&self, _log_folding: usize) -> Option<Vec<[u8; 32]>> {
        match self.0 {}
    }

    pub(crate) fn cosets<F>(
        &self,
        _indices: &[usize],
        _num_leaves: usize,
        _block: usize,
    ) -> Option<Vec<Vec<math::field::element::FieldElement<F>>>>
    where
        F: math::field::traits::IsField + 'static,
    {
        match self.0 {}
    }

    pub(crate) fn first<F>(&self) -> Option<math::field::element::FieldElement<F>>
    where
        F: math::field::traits::IsField + 'static,
    {
        match self.0 {}
    }
}
