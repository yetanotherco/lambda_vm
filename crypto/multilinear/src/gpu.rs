//! Device dispatch for the multilinear path.
//!
//! Every entry point here returns `None` when the device declines — a field
//! the kernels do not cover, a size below the launch threshold, a kill switch,
//! or any CUDA error — and the caller runs the host path. A dispatch that
//! succeeded is counted, so a bench can tell a GPU number from a CPU one
//! wearing its label.
//!
//! There is no device yet: this is the half that declines, so every entry point
//! returns `None` unconditionally and the counters stay at zero. The half that
//! does the work lands last, and until it does the whole path is host-only —
//! which is how it is meant to be read.

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

/// A sumcheck's round proofs, the challenges they drew, and what every slot
/// was bound to — the factors are folded where they lie, so their values at
/// the sumcheck's point are already there when the rounds end.
/// What a resident sumcheck's rounds on device leave: the rounds, the point
/// they drew, and **every** factor — resident and not — as the last fold left
/// it. One value each when the device ran the cube out, a cube when it stopped
/// at the crossover for the caller to finish.
type ResidentRounds<E> = (
    Vec<crate::sumcheck::RoundProof<E>>,
    Vec<math::field::element::FieldElement<E>>,
    Vec<crate::mle::Mle<E>>,
);

type SumcheckRounds<E> = (
    Vec<crate::sumcheck::RoundProof<E>>,
    Vec<math::field::element::FieldElement<E>>,
    Vec<crate::mle::Mle<E>>,
);

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
///
/// The slot file is `slots * 24 * threads` bytes and the scratch budget is
/// fixed, so a wider program buys fewer threads. This is where that stops
/// being a trade: at 8192 live values a single block of 256 already wants
/// 48 MiB, and a launch of one block is not a launch.
///
/// It is a cliff, not a dial. The real AIRs peak near a thousand — the widest
/// precompile lowers to 1036 — and a cap below that sends the tables with the
/// *most* work per row to the host, which is where they cost the most.
pub const MAX_SLOTS: usize = 8192;

pub fn reserve_room(_bytes: u64) -> Option<DeviceRoom> {
    None
}

/// A promise no device made. Never constructed.
#[derive(Debug)]
pub struct DeviceRoom(std::convert::Infallible);

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

pub(crate) fn prove_sumcheck<E>(
    _polys: &[crate::mle::Mle<E>],
    _program: &crate::program::Program<E>,
    _degree: usize,
    _host_cube: usize,
    _challenge: impl FnMut(
        &[math::field::element::FieldElement<E>],
    ) -> math::field::element::FieldElement<E>,
) -> Option<Result<SumcheckRounds<E>, crate::Error>>
where
    E: math::field::traits::IsField + 'static,
{
    None
}

pub(crate) fn prove_sumcheck_resident<E>(
    _resident: &DeviceFactors,
    _extra: &[crate::mle::Mle<E>],
    _program: &crate::program::Program<E>,
    _degree: usize,
    _challenge: impl FnMut(
        &[math::field::element::FieldElement<E>],
    ) -> math::field::element::FieldElement<E>,
) -> Option<Result<ResidentRounds<E>, crate::Error>>
where
    E: math::field::traits::IsField + 'static,
{
    None
}

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

pub(crate) fn commit_tree_ext3<F>(
    _codeword: &[math::field::element::FieldElement<F>],
    _log_folding: usize,
) -> Option<Vec<[u8; 32]>>
where
    F: math::field::traits::IsField + 'static,
{
    None
}

pub(crate) fn evaluate_many_base<F, E>(
    _columns: &[crate::mle::Mle<F>],
    _point: &[math::field::element::FieldElement<E>],
    _resident: Option<(&ResidentColumns, usize)>,
) -> Option<Vec<math::field::element::FieldElement<E>>>
where
    F: math::field::traits::IsField + 'static,
    E: math::field::traits::IsField + 'static,
{
    None
}

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

/// A device tree the build declined to make. Never constructed.
#[derive(Debug)]
pub struct DeviceTree(std::convert::Infallible);

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
        _tail: usize,
        _challenge: impl FnMut(
            &[math::field::element::FieldElement<E>],
        ) -> math::field::element::FieldElement<E>,
    ) -> Option<Result<LayerRounds<E>, crate::Error>>
    where
        E: math::field::traits::IsField + 'static,
    {
        match self.0 {}
    }

    pub(crate) fn layer_to_host<E>(
        &self,
        _layer: usize,
    ) -> Option<(crate::mle::Mle<E>, crate::mle::Mle<E>)>
    where
        E: math::field::traits::IsField + 'static,
    {
        match self.0 {}
    }
}

/// What a layer's rounds on device leave: the rounds themselves, the point
/// they drew, and the five factors as the last fold left them.
///
/// The factors are one value each when the device ran the layer out, and a
/// cube when it stopped at the crossover for the host to finish — the caller
/// carries on from them either way.
pub type LayerRounds<E> = (
    Vec<crate::sumcheck::RoundProof<E>>,
    Vec<math::field::element::FieldElement<E>>,
    Vec<crate::mle::Mle<E>>,
);

pub(crate) fn build_tree<E>(_p: &crate::mle::Mle<E>, _q: &crate::mle::Mle<E>) -> Option<DeviceTree>
where
    E: math::field::traits::IsField + 'static,
{
    None
}

/// One that could not be made. Never constructed.
pub struct ResidentColumns(std::convert::Infallible);

impl std::fmt::Debug for ResidentColumns {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("ResidentColumns")
    }
}

pub fn upload_columns<F>(_columns: &[&crate::mle::Mle<F>]) -> Option<ResidentColumns>
where
    F: math::field::traits::IsField + 'static,
{
    None
}

/// Factors a build declined to upload. Never constructed.
#[derive(Debug)]
pub struct DeviceFactors(std::convert::Infallible);

pub fn input_layer_tree<E>(
    _factors: std::sync::Arc<DeviceFactors>,
    _numerators: Vec<crate::program::Program<E>>,
    _denominators: Vec<crate::program::Program<E>>,
) -> Option<DeviceTree>
where
    E: math::field::traits::IsField + 'static,
{
    None
}

/// Factors a build declined to upload. Never constructed.
pub struct OpeningFactors(std::convert::Infallible);

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

pub(crate) fn open_shared<F, E>(
    _message: &crate::whir_chain::Stacked<'_, F>,
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

/// A codeword a commit declined to keep. Never constructed.
pub struct DeviceCodeword(std::convert::Infallible);

impl std::fmt::Debug for DeviceCodeword {
    fn fmt(&self, _: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self.0 {}
    }
}

pub(crate) fn commit_resident(
    _store: &ResidentColumns,
    _parts: &[(usize, usize)],
    _log_evals: usize,
    _log_blowup: usize,
    _log_folding: usize,
    _transient: bool,
) -> Option<(DeviceCodeword, [u8; 32])> {
    None
}

pub(crate) fn commit_parts<F>(
    _parts: &[(&crate::mle::Mle<F>, usize)],
    _log_evals: usize,
    _log_blowup: usize,
    _log_folding: usize,
    _transient: bool,
) -> Option<(DeviceCodeword, [u8; 32])>
where
    F: math::field::traits::IsField + 'static,
{
    None
}

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

    pub(crate) fn commit(&self, _log_folding: usize) -> Option<[u8; 32]> {
        match self.0 {}
    }

    pub(crate) fn paths(
        &self,
        _log_folding: usize,
        _indices: &[usize],
    ) -> Option<Vec<Vec<[u8; 32]>>> {
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
