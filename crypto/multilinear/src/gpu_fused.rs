//! D-ARGUE stage 1 on the card (`thoughts/zf/gap2/fix2/D-ARGUE.md` §4.2): a
//! table's zerocheck rounds with the bus as one column (S1-2), the weights
//! pulled out (S1-4), rounds 0 and 1 from one base-field pass (S1-5) and
//! integer nodes (S1-1) — every message today's.
//!
//! [`fused`](crate::fused) is the host reference and this is its device twin,
//! round for round: the same parts (`T`, `U`, `A`, `B`), the same messages out
//! of them ([`fused::message`]). The kernels are `math_cuda::argue_fused`'s.
//!
//! On for every table whose constraint part runs in the base field; a test's
//! [`force_argue_fused`] off is today's rounds exactly.
//! A table it declines runs today's rounds, counted. `LAMBDA_VM_ARGUE_FUSED_XCHECK=1`
//! runs today's rounds after it over the same factors with the same
//! challenges, refuses the prove at the first message that differs, times
//! both, and checks the grid's corners.
//!
//! The default was turned on by its A/B on the block (FAST jobs 274 and 270,
//! `thoughts/zf/gap2/fix2/I-ARGUE.md` §4.5–4.6): with integer nodes, the argue
//! 11.70 → 8.75 s and the whole run 2.80 s faster at 9e2728955, every arm
//! proved and verified on the record's identities, and the cross-check arm —
//! 390 fused tables, every message compared with today's — clean.

use std::sync::atomic::{AtomicU8, AtomicU64, Ordering};

use math::field::{element::FieldElement, traits::IsField};

use crate::{fused::Constraints, logup::Interaction, program::Op};

/// The `ACC` op and its no-selector marker. MUST stay in sync with
/// `crypto/math-cuda/kernels/sumcheck.cu`.
pub mod op {
    pub const ACC: u32 = 6;
    pub const NO_SELECTOR: u32 = u32::MAX;
}

/// What the fused rounds need beside the factors and the batch: the
/// constraint part in the base field, its batching powers, the bus, and the
/// two points.
pub struct FusedInput<'a, B: IsField, E: IsField> {
    pub constraints: &'a Constraints<B>,
    pub betas: &'a [FieldElement<E>],
    pub interactions: &'a [Interaction<E>],
    /// The GKR input-layer claim's point; its last `num_vars` coordinates are `ρ`.
    pub claim_point: &'a [FieldElement<E>],
    pub r: &'a [FieldElement<E>],
    /// The table's base columns on the card as factors, with no lift
    /// (`gpu::argue_no_lift`, D-BATCH M1-2): the rounds read them when no
    /// lifted factors were made.
    pub columns: Option<&'a crate::gpu::ColumnFactors>,
    /// A second bus at its own claim point (a table's local lookups): its two
    /// rules are the batch's fourth and fifth, at `λ³` and `λ⁴`.
    pub second: Option<SecondInput<'a, E>>,
}

/// A second bus's interactions and its GKR input-layer claim point (whose
/// last `num_vars` coordinates are its `ρ'`).
pub struct SecondInput<'a, E: IsField> {
    pub interactions: &'a [Interaction<E>],
    pub claim_point: &'a [FieldElement<E>],
}

/// What the fused rounds read their factors through: the lifted factors, or
/// the base columns where they lie (D-BATCH M1-2).
#[cfg(feature = "cuda")]
#[derive(Clone, Copy)]
pub(crate) enum FusedSource<'a> {
    Lifted(&'a crate::gpu::DeviceFactors),
    Columns(&'a crate::gpu::ColumnFactors),
}

#[cfg(feature = "cuda")]
impl<'a> FusedSource<'a> {
    /// `(width, rows)`: the factors and the cube they span.
    pub(crate) fn shape(self) -> (usize, usize) {
        let view = self.view();
        (view.width, view.rows)
    }

    fn view(self) -> math_cuda::sumcheck::FactorView<'a> {
        match self {
            Self::Lifted(factors) => factors.inner().view(),
            Self::Columns(columns) => columns.view(),
        }
    }
}

/// The constraint part lowered for both kernels: the base DAG with an `ACC`
/// step after each root (`acc += β[i]·root·selector`), slots reused past
/// their last read as [`crate::gpu::lower`] does.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LoweredFused {
    pub nodes: Vec<u64>,
    pub base_consts: Vec<u64>,
    pub ext_consts: Vec<u64>,
    pub num_slots: usize,
}

#[cfg(feature = "cuda")]
impl LoweredFused {
    #[doc(hidden)]
    pub fn device_program<'a>(
        &'a self,
        betas: &'a [u64],
    ) -> math_cuda::argue_fused::FusedProgram<'a> {
        math_cuda::argue_fused::FusedProgram {
            nodes: &self.nodes,
            base_consts: &self.base_consts,
            ext_consts: &self.ext_consts,
            betas,
            num_slots: self.num_slots,
        }
    }
}

#[doc(hidden)]
pub const LOW_GRID_POINTS: [(u32, u32); 5] = [(2, 0), (2, 1), (0, 2), (1, 2), (2, 2)];

/// Adds the quadratic part to the cubic grid. Its four corners are zero on
/// trace rows. Extrapolating from 0, 1 and 2 to 3 uses weights 1, -3 and 3.
pub fn add_low_grid<E: IsField>(t: &mut [[FieldElement<E>; 4]; 4], low: &[FieldElement<E>; 5]) {
    let thrice = |x: &FieldElement<E>| x + x + x;
    let at_three =
        |x0: &FieldElement<E>, x1: &FieldElement<E>, x2: &FieldElement<E>| x0 + thrice(&(x2 - x1));
    let [t20, t21, t02, t12, t22] = low;
    for (&(a, b), value) in LOW_GRID_POINTS.iter().zip(low) {
        t[a as usize][b as usize] += value.clone();
    }
    let t03 = thrice(t02);
    let t13 = thrice(t12);
    let t23 = at_three(t20, t21, t22);
    t[3][3] += at_three(&t03, &t13, &t23);
    t[0][3] += t03;
    t[1][3] += t13;
    t[2][3] += t23;
    t[3][0] += thrice(t20);
    t[3][1] += thrice(t21);
    t[3][2] += at_three(t02, t12, t22);
}

/// A base-field element's canonical u64, or `None` off Goldilocks.
fn base_raw<B: IsField + 'static>(value: &FieldElement<B>) -> Option<u64> {
    use math::field::goldilocks::GoldilocksField as Gl;
    if std::any::TypeId::of::<B>() != std::any::TypeId::of::<Gl>() {
        return None;
    }
    // SAFETY: `B == Gl`, checked above.
    let gl: FieldElement<Gl> =
        unsafe { core::mem::transmute_copy::<FieldElement<B>, FieldElement<Gl>>(value) };
    Some(gl.canonical_u64())
}

/// Lowers `constraints` with its `ACC` steps. `None` off Goldilocks or past
/// [`crate::gpu::MAX_SLOTS`].
pub fn lower_fused<B: IsField + 'static>(constraints: &Constraints<B>) -> Option<LoweredFused> {
    lower_fused_part(constraints, None)
}

/// Lower the quadratic-or-lower and cubic roots of a degree-three program.
/// An ACC keeps its original beta index; each part only walks live steps.
pub fn lower_fused_grid_split<B: IsField + 'static>(
    constraints: &Constraints<B>,
) -> Option<[LoweredFused; 2]> {
    if constraints.degree() != 3 {
        return None;
    }
    let mut degrees: Vec<usize> = Vec::with_capacity(constraints.steps().len());
    for step in constraints.steps() {
        degrees.push(match *step {
            Op::Fixed(_) => 0,
            Op::Var(_) => 1,
            Op::Neg(a) => degrees[a as usize],
            Op::Add(a, b) | Op::Sub(a, b) => degrees[a as usize].max(degrees[b as usize]),
            Op::Mul(a, b) => degrees[a as usize].saturating_add(degrees[b as usize]),
        });
    }
    let (low, high): (Vec<_>, Vec<_>) = (0..constraints.roots().len()).partition(|&i| {
        let degree = degrees[constraints.roots()[i] as usize];
        degree <= 2 - usize::from(constraints.selectors()[i].is_some())
    });
    if low.is_empty() || high.is_empty() {
        return None;
    }
    Some([
        lower_fused_part(constraints, Some(&low))?,
        lower_fused_part(constraints, Some(&high))?,
    ])
}

fn lower_fused_part<B: IsField + 'static>(
    constraints: &Constraints<B>,
    selected: Option<&[usize]>,
) -> Option<LoweredFused> {
    enum Item {
        Step(usize),
        Acc(usize),
    }
    let steps = constraints.steps();
    let roots = constraints.roots();
    let selectors = constraints.selectors();
    // Each root's ACC right after the step that makes it.
    let mut after: Vec<Vec<usize>> = vec![Vec::new(); steps.len()];
    let mut live = selected.map(|_| vec![false; steps.len()]);
    for i in 0..selected.map_or(roots.len(), <[usize]>::len) {
        let i = selected.map_or(i, |indices| indices[i]);
        let root = roots[i] as usize;
        after.get_mut(root)?.push(i);
        if let Some(live) = &mut live {
            live[root] = true;
        }
    }
    if let Some(live) = &mut live {
        for (s, step) in steps.iter().enumerate().rev() {
            if live[s] {
                match *step {
                    Op::Fixed(_) | Op::Var(_) => {}
                    Op::Neg(a) => live[a as usize] = true,
                    Op::Add(a, b) | Op::Sub(a, b) | Op::Mul(a, b) => {
                        live[a as usize] = true;
                        live[b as usize] = true;
                    }
                }
            }
        }
    }
    let mut items = Vec::with_capacity(steps.len() + roots.len());
    for (s, accs) in after.iter().enumerate() {
        if live.as_ref().is_some_and(|live| !live[s]) {
            continue;
        }
        items.push(Item::Step(s));
        items.extend(accs.iter().map(|&i| Item::Acc(i)));
    }
    // Each value's last reader, by item position.
    let mut last_use = vec![0usize; steps.len()];
    for (at, item) in items.iter().enumerate() {
        match *item {
            Item::Step(s) => match steps[s] {
                Op::Fixed(_) | Op::Var(_) => {}
                Op::Neg(a) => last_use[a as usize] = at,
                Op::Add(a, b) | Op::Sub(a, b) | Op::Mul(a, b) => {
                    last_use[a as usize] = at;
                    last_use[b as usize] = at;
                }
            },
            Item::Acc(i) => last_use[roots[i] as usize] = at,
        }
    }
    let mut slot_of = vec![u32::MAX; steps.len()];
    let mut free: Vec<u32> = Vec::new();
    let mut num_slots = 0usize;
    let mut nodes = Vec::with_capacity(items.len() * 2);
    let mut base_consts = Vec::new();
    let mut ext_consts = Vec::new();
    let push = |nodes: &mut Vec<u64>, op: u32, a: u32, b: u32, res: u32| {
        nodes.push(u64::from(op) | (u64::from(a) << 32));
        nodes.push(u64::from(b) | (u64::from(res) << 32));
    };
    for (at, item) in items.iter().enumerate() {
        match *item {
            Item::Acc(i) => {
                let value = slot_of[roots[i] as usize];
                let selector = match selectors[i] {
                    Some(slot) => u32::try_from(slot).ok()?,
                    None => op::NO_SELECTOR,
                };
                push(&mut nodes, op::ACC, value, selector, i as u32);
                if last_use[roots[i] as usize] == at {
                    free.push(value);
                }
            }
            Item::Step(s) => {
                let (tag, a, b, operands) = match steps[s] {
                    Op::Fixed(ref c) => {
                        let raw = base_raw(c)?;
                        let k = base_consts.len() as u32;
                        base_consts.push(raw);
                        ext_consts.extend_from_slice(&[raw, 0, 0]);
                        (crate::gpu::op::FIXED, k, 0, [None, None])
                    }
                    Op::Var(slot) => (crate::gpu::op::VAR, slot, 0, [None, None]),
                    Op::Neg(a) => (crate::gpu::op::NEG, slot_of[a as usize], 0, [Some(a), None]),
                    Op::Add(a, b) | Op::Sub(a, b) | Op::Mul(a, b) => {
                        let tag = match steps[s] {
                            Op::Add(..) => crate::gpu::op::ADD,
                            Op::Sub(..) => crate::gpu::op::SUB,
                            _ => crate::gpu::op::MUL,
                        };
                        (
                            tag,
                            slot_of[a as usize],
                            slot_of[b as usize],
                            [Some(a), Some(b)],
                        )
                    }
                };
                if let Some(x) = operands[0].filter(|x| last_use[*x as usize] == at) {
                    free.push(slot_of[x as usize]);
                }
                if let Some(y) =
                    operands[1].filter(|y| last_use[*y as usize] == at && operands[0] != Some(*y))
                {
                    free.push(slot_of[y as usize]);
                }
                let res = free.pop().unwrap_or_else(|| {
                    num_slots += 1;
                    (num_slots - 1) as u32
                });
                if num_slots > crate::gpu::MAX_SLOTS {
                    return None;
                }
                slot_of[s] = res;
                push(&mut nodes, tag, a, b, res);
                // A value nothing reads — a root is read by its ACC, and no
                // step reads itself — frees its slot at once.
                if last_use[s] == 0 {
                    free.push(res);
                }
            }
        }
    }
    Some(LoweredFused {
        nodes,
        base_consts,
        ext_consts,
        num_slots: num_slots.max(1),
    })
}

/// The kernels' walk of a lowered blob, on the host: `Σ β_i·sel_i·root_i` at
/// one row whose factors are `values` (base field), each step in the base
/// field and each `ACC` a base-by-extension multiply — what `zc_grid01` does at
/// a grid point. For tests: it is how a lowering is checked against the DAG
/// it came from without a card.
pub fn run_lowered_host<B, E>(
    lowered: &LoweredFused,
    betas: &[FieldElement<E>],
    values: &[FieldElement<B>],
) -> FieldElement<E>
where
    B: IsField + math::field::traits::IsSubFieldOf<E> + 'static,
    E: IsField,
{
    let mut slots = vec![FieldElement::<B>::zero(); lowered.num_slots];
    let mut acc = FieldElement::<E>::zero();
    for node in lowered.nodes.chunks_exact(2) {
        let (tag, a) = ((node[0] & 0xFFFF_FFFF) as u32, (node[0] >> 32) as u32);
        let (b, res) = ((node[1] & 0xFFFF_FFFF) as u32, (node[1] >> 32) as u32);
        let v = |s: u32| slots[s as usize].clone();
        let value = match tag {
            t if t == crate::gpu::op::FIXED => {
                FieldElement::<B>::from(lowered.base_consts[a as usize])
            }
            t if t == crate::gpu::op::VAR => values[a as usize].clone(),
            t if t == crate::gpu::op::ADD => v(a) + v(b),
            t if t == crate::gpu::op::SUB => v(a) - v(b),
            t if t == crate::gpu::op::MUL => v(a) * v(b),
            t if t == crate::gpu::op::NEG => -v(a),
            _ => {
                let term = if b == op::NO_SELECTOR {
                    v(a)
                } else {
                    v(a) * values[b as usize].clone()
                };
                acc += &term * &betas[res as usize];
                continue;
            }
        };
        slots[res as usize] = value;
    }
    acc
}

// ── knobs, counters, faults ──────────────────────────────────────────────────

fn env_is(name: &str) -> bool {
    std::env::var(name).is_ok_and(|v| v == "1" || v == "true")
}

/// Walk degree-three constraint grids as separate quadratic and cubic parts.
/// On by default. A trimmed `LAMBDA_VM_ZC_GRID_SPLIT=0` disables the split.
/// Corner checks and grids that retain the corners use the whole program.
pub fn zc_grid_split() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| {
        zc_grid_split_from_env(std::env::var("LAMBDA_VM_ZC_GRID_SPLIT").ok().as_deref())
    })
}

fn zc_grid_split_from_env(value: Option<&str>) -> bool {
    value.is_none_or(|v| v.trim() != "0")
}

static FUSED_FORCED: AtomicU8 = AtomicU8::new(0);
static FUSED_XCHECK_FORCED: AtomicU8 = AtomicU8::new(0);

fn forced(flag: &AtomicU8, env: impl FnOnce() -> bool) -> bool {
    match flag.load(Ordering::Relaxed) {
        1 => false,
        2 => true,
        _ => env(),
    }
}

fn store_forced(flag: &AtomicU8, on: Option<bool>) {
    flag.store(
        match on {
            None => 0,
            Some(false) => 1,
            Some(true) => 2,
        },
        Ordering::Relaxed,
    );
}

/// Whether the zerocheck runs the fused rounds: on, unless a test forces them
/// off ([`force_argue_fused`]).
pub fn argue_fused() -> bool {
    forced(&FUSED_FORCED, || true)
}

#[doc(hidden)]
pub fn force_argue_fused(on: Option<bool>) {
    store_forced(&FUSED_FORCED, on);
}

/// Whether the fused rounds are cross-checked against today's
/// (`LAMBDA_VM_ARGUE_FUSED_XCHECK=1`; default off). Read once.
pub fn argue_fused_xcheck() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    forced(&FUSED_XCHECK_FORCED, || {
        *ON.get_or_init(|| env_is("LAMBDA_VM_ARGUE_FUSED_XCHECK"))
    })
}

#[doc(hidden)]
pub fn force_argue_fused_xcheck(on: Option<bool>) {
    store_forced(&FUSED_XCHECK_FORCED, on);
}

/// A program of at least this many steps keeps its rounds on the card past
/// the host's crossover: the host walks it
/// through the interpreter once for every index pair and node of a round,
/// while a card round costs about one walk whatever the cube.
pub const DEVICE_TAIL_MIN_STEPS: usize = 1000;

/// The cube such a program hands the host instead of today's 32: on an RTX
/// 5090 a KECCAK_RND round costs the card ≈ 0.5 ms and the host ≈ 0.24 ms an
/// index pair, so the card's round is the cheaper one down to a cube of 8
/// (4 pairs) and the host's from a cube of 4 (2 pairs).
pub const DEVICE_TAIL_CUBE: usize = 1 << 2;

static TAIL_FORCED: AtomicU8 = AtomicU8::new(0);
static TAIL_FORCED_CUBE: AtomicU64 = AtomicU64::new(0);

/// Where the fused rounds hand a long program to the host: `None` at the
/// host's crossover, or `(min_steps, cube)` — a program of at least
/// `min_steps` steps stops at `cube` instead. The same messages either way, so
/// the same proof. `(DEVICE_TAIL_MIN_STEPS, DEVICE_TAIL_CUBE)` unless a test
/// forces another mode ([`force_zc_device_tail`]).
pub fn zc_device_tail() -> Option<(usize, usize)> {
    match TAIL_FORCED.load(Ordering::Relaxed) {
        1 => None,
        2 => Some((0, TAIL_FORCED_CUBE.load(Ordering::Relaxed) as usize)),
        _ => Some((DEVICE_TAIL_MIN_STEPS, DEVICE_TAIL_CUBE)),
    }
}

/// Forces a mode: `None` back to the default, `Some(None)` the crossover,
/// `Some(Some(cube))` every program, whatever its length, down to `cube` (a
/// power of two, at least 2) — a test's way to the card's smallest rounds on
/// short programs.
#[doc(hidden)]
pub fn force_zc_device_tail(mode: Option<Option<usize>>) {
    if let Some(Some(cube)) = mode {
        assert!(
            cube >= 2 && cube.is_power_of_two(),
            "the host takes a cube of at least 2"
        );
        TAIL_FORCED_CUBE.store(cube as u64, Ordering::Relaxed);
    }
    TAIL_FORCED.store(
        match mode {
            None => 0,
            Some(None) => 1,
            Some(Some(_)) => 2,
        },
        Ordering::Relaxed,
    );
}

/// ⛔ Faults and inert checks, for the negative controls and the mutations
/// that show each check is load-bearing. Never armed outside a test.
#[doc(hidden)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct FusedFaults {
    /// The card's first bus coefficient off by one.
    pub bus: bool,
    /// Not a fault: the grid's corners summed into `T` rather than taken as
    /// zero — what random columns, which break every AIR, need to send
    /// today's messages. Overrides the corner check.
    pub keep_corners: bool,
    /// The cross-check compares nothing.
    pub xcheck_inert: bool,
    /// The corner check refuses nothing.
    pub corners_inert: bool,
}

thread_local! {
    // Per thread, not per process: the fused rounds are on by default, so a
    // fault armed for one test must not reach a proof another test runs beside
    // it. A table's argument runs on the thread that calls it (the table loop
    // is serial), which is the test's own.
    static FAULTS: std::cell::Cell<u8> = const { std::cell::Cell::new(0) };
}

/// Arms `faults` for the calling thread's proofs; `FusedFaults::default()`
/// disarms.
#[doc(hidden)]
pub fn force_fused_faults(faults: FusedFaults) {
    FAULTS.with(|f| {
        f.set(
            u8::from(faults.bus)
                | (u8::from(faults.keep_corners) << 1)
                | (u8::from(faults.xcheck_inert) << 2)
                | (u8::from(faults.corners_inert) << 3),
        )
    });
}

#[cfg_attr(not(feature = "cuda"), allow(dead_code))]
pub(crate) fn faults() -> FusedFaults {
    let f = FAULTS.with(std::cell::Cell::get);
    FusedFaults {
        bus: f & 1 != 0,
        keep_corners: f & 2 != 0,
        xcheck_inert: f & 4 != 0,
        corners_inert: f & 8 != 0,
    }
}

static SESSIONS: AtomicU64 = AtomicU64::new(0);
static DECLINES: AtomicU64 = AtomicU64::new(0);
static HOST_ZEROCHECKS: AtomicU64 = AtomicU64::new(0);
static XCHECKS: AtomicU64 = AtomicU64::new(0);
static CORNER_CHECKS: AtomicU64 = AtomicU64::new(0);

/// Tables whose zerocheck ran the fused rounds.
pub fn fused_sessions() -> u64 {
    SESSIONS.load(Ordering::Relaxed)
}

/// Tables the fused rounds declined before round 0 (today's ran).
pub fn fused_declines() -> u64 {
    DECLINES.load(Ordering::Relaxed)
}

/// Zerochecks that fell back from device setup and ran their full rounds on
/// the host. The scheduled host tail of a device session is separate.
pub fn host_zerochecks() -> u64 {
    HOST_ZEROCHECKS.load(Ordering::Relaxed)
}

#[cfg(feature = "cuda")]
pub(crate) struct ZerocheckDecline {
    reason: std::borrow::Cow<'static, str>,
    asked: u64,
    free: Option<u64>,
}

#[cfg(feature = "cuda")]
impl ZerocheckDecline {
    fn new(reason: impl Into<std::borrow::Cow<'static, str>>, asked: u64) -> Self {
        Self {
            reason: reason.into(),
            asked,
            free: None,
        }
    }

    pub(crate) fn no_factors() -> Self {
        Self::new("device factors unavailable", 0)
    }

    pub(crate) fn note_host(self, vars: usize) {
        let free = self.free.unwrap_or_else(|| {
            crate::gpu::reserve_budget().saturating_sub(crate::gpu::ledger_reserved())
        });
        let count = HOST_ZEROCHECKS.fetch_add(1, Ordering::Relaxed) + 1;
        eprintln!(
            "[argue] zerocheck on the host: {} · 2^{vars} rows · asked {} B · ledger free {free} B · host zerochecks {count}",
            self.reason, self.asked,
        );
    }
}

/// A small cube starts on the host by policy.
#[cfg(feature = "cuda")]
pub(crate) fn worth_fused_rounds(vars: usize) -> bool {
    vars >= crate::HOST_CUBE_COMPILED.trailing_zeros() as usize + 2
}

#[cfg(feature = "cuda")]
thread_local! {
    static OPENING_RELEASE: std::cell::RefCell<Option<std::sync::mpsc::Receiver<()>>> =
        const { std::cell::RefCell::new(None) };
}

/// During an argue beside an opening on the same card, a refused zerocheck
/// reservation can wait for that opening to drop its working set, then retry.
/// The opening drops the sender even if it fails or panics.
#[doc(hidden)]
pub fn with_opening_release<R>(
    release: Option<std::sync::mpsc::Receiver<()>>,
    argue: impl FnOnce() -> R,
) -> R {
    #[cfg(feature = "cuda")]
    let _restore = {
        struct Restore(Option<std::sync::mpsc::Receiver<()>>);
        impl Drop for Restore {
            fn drop(&mut self) {
                OPENING_RELEASE.with(|slot| slot.replace(self.0.take()));
            }
        }
        Restore(OPENING_RELEASE.with(|slot| slot.replace(release)))
    };
    #[cfg(not(feature = "cuda"))]
    drop(release);
    argue()
}

#[cfg(feature = "cuda")]
fn reserve_zerocheck<R>(
    mut reserve: impl FnMut() -> Result<R, u64>,
    fits_budget: impl FnOnce() -> bool,
) -> Result<R, u64> {
    let free = match reserve() {
        Ok(room) => return Ok(room),
        Err(free) => free,
    };
    if fits_budget()
        && let Some(release) = OPENING_RELEASE.with(|slot| slot.take())
    {
        let _ = release.recv();
        return reserve();
    }
    Err(free)
}

/// Fused tables whose every message today's rounds confirmed.
pub fn fused_xchecks() -> u64 {
    XCHECKS.load(Ordering::Relaxed)
}

/// Fused tables whose grid corners were computed and found zero.
pub fn fused_corner_checks() -> u64 {
    CORNER_CHECKS.load(Ordering::Relaxed)
}

#[cfg_attr(not(feature = "cuda"), allow(dead_code))]
pub(crate) fn note_decline() {
    DECLINES.fetch_add(1, Ordering::Relaxed);
}

#[cfg_attr(not(feature = "cuda"), allow(dead_code))]
pub(crate) fn note_xcheck() {
    XCHECKS.fetch_add(1, Ordering::Relaxed);
}

#[doc(hidden)]
pub fn reset_fused_counters() {
    for c in [
        &SESSIONS,
        &DECLINES,
        &HOST_ZEROCHECKS,
        &XCHECKS,
        &CORNER_CHECKS,
    ] {
        c.store(0, Ordering::Relaxed);
    }
}

/// One fused table's timings, for its log line.
#[derive(Clone, Copy, Debug, Default)]
pub struct FusedTimes {
    pub grid: f64,
    pub grid_split: bool,
    pub fold2: f64,
    pub rounds: f64,
    pub total: f64,
    /// The table's signature on the log line (a committed table has no name
    /// here): its roots, its bus column's terms, the constraint walk's slots.
    pub roots: usize,
    pub terms: usize,
    pub slots: usize,
}

static DEVICE_TAILS: AtomicU64 = AtomicU64::new(0);

/// Fused tables whose rounds stayed on the card past today's crossover
/// ([`zc_device_tail`]).
pub fn device_tail_sessions() -> u64 {
    DEVICE_TAILS.load(Ordering::Relaxed)
}

/// Where a thread's fused tables spent their later rounds since the last
/// [`take_tail`], short programs at `[0]` and those of at least
/// [`DEVICE_TAIL_MIN_STEPS`] steps at `[1]` (in every mode): the batched
/// argue's log line under the split.
#[derive(Clone, Copy, Debug, Default)]
pub struct TailTimes {
    /// Fused tables.
    pub tables: [u32; 2],
    /// Of them, those whose rounds stayed on the card past today's crossover.
    pub on_card: [u32; 2],
    /// The card's later rounds and their folds, seconds as the host saw them.
    pub device: [f64; 2],
    /// The rounds the host ran over what the card folded, and their seconds
    /// (each round's evaluations and its fold).
    pub host_rounds: [u32; 2],
    pub host: [f64; 2],
}

thread_local! {
    static TAIL: std::cell::Cell<TailTimes> = std::cell::Cell::new(TailTimes::default());
}

#[cfg_attr(not(feature = "cuda"), allow(dead_code))]
pub(crate) fn note_tail(note: impl FnOnce(&mut TailTimes)) {
    TAIL.with(|cell| {
        let mut tail = cell.get();
        note(&mut tail);
        cell.set(tail);
    });
}

/// The calling thread's [`TailTimes`] since the last call, reset.
pub fn take_tail() -> TailTimes {
    TAIL.with(|cell| cell.replace(TailTimes::default()))
}

// ── the rounds ───────────────────────────────────────────────────────────────

/// The device rounds' result: the messages, the challenges, and every factor
/// — the trace's, then the two weights — as the last device fold left it.
#[cfg(feature = "cuda")]
pub(crate) type FusedRounds<E> = (
    Vec<crate::sumcheck::RoundProof<E>>,
    Vec<FieldElement<E>>,
    Vec<crate::mle::Mle<E>>,
    FusedTimes,
);

/// A table's zerocheck rounds the fused way, down to the host crossover.
///
/// `lambdas` are the batch's weights, `claim` its combined claim, `degree`
/// the batch's (`d_C + 1`). `challenge` absorbs a message and draws the next
/// challenge, as today's device rounds call it.
///
/// A decline comes before the first message, when the caller can still fall back.
///
/// The rounds are [`FusedStepper`]'s, driven here one after the other; the
/// batched argue drives the same stepper in lockstep with other tables'.
#[cfg(feature = "cuda")]
#[allow(clippy::too_many_arguments)]
pub(crate) fn prove_fused<B, E>(
    source: FusedSource<'_>,
    input: &FusedInput<'_, B, E>,
    lambdas: &[FieldElement<E>],
    claim: &FieldElement<E>,
    degree: usize,
    check_corners: bool,
    mut challenge: impl FnMut(&[FieldElement<E>]) -> FieldElement<E>,
) -> Result<Result<FusedRounds<E>, crate::Error>, ZerocheckDecline>
where
    B: IsField + math::field::traits::IsSubFieldOf<E> + 'static,
    E: IsField + 'static,
    FieldElement<E>: Send + Sync,
{
    let mut stepper = match FusedStepper::new(source, input, lambdas, claim, degree, check_corners)?
    {
        Ok(stepper) => stepper,
        Err(error) => return Ok(Err(error)),
    };
    let there = stepper.device_rounds();
    let mut rounds = Vec::with_capacity(there);
    let mut point = Vec::with_capacity(there);
    for _ in 0..there {
        let g = match stepper.message() {
            Ok(g) => g,
            Err(error) => return Ok(Err(error)),
        };
        let s = challenge(&g);
        if let Err(error) = stepper.bind(&s) {
            return Ok(Err(error));
        }
        rounds.push(crate::sumcheck::RoundProof { evaluations: g });
        point.push(s);
    }
    let (factors, times) = match stepper.into_factors() {
        Ok(done) => done,
        Err(error) => return Ok(Err(error)),
    };
    Ok(Ok((rounds, point, factors, times)))
}

/// A table's fused zerocheck rounds, stepped by its caller: the message of the
/// round under way, then the challenge to bind it with — so several tables'
/// rounds can share one challenge (the batched argue, D-BATCH B-3) and one
/// table's can run alone ([`prove_fused`]) with the same messages.
///
/// Rounds 0 and 1 come from the grid pass (S1-5), the later ones from the
/// Gruen rounds (S1-4) until today's host crossover; [`Self::into_factors`]
/// then hands back every factor as the last fold left it, for the host tail.
#[cfg(feature = "cuda")]
#[doc(hidden)]
pub struct FusedStepper<'f, E: IsField> {
    session: math_cuda::argue_fused::FusedZerocheck<'f>,
    r: Vec<FieldElement<E>>,
    rho: Vec<FieldElement<E>>,
    d: usize,
    big: FieldElement<E>,
    walked: Vec<u32>,
    there: usize,
    /// Today's crossover, which `there` passes under the device tail.
    crossover: usize,
    /// The lowered program's steps.
    steps: usize,
    t: Vec<Vec<FieldElement<E>>>,
    u: [[FieldElement<E>; 2]; 2],
    claim: FieldElement<E>,
    e_r: FieldElement<E>,
    e_rho: FieldElement<E>,
    /// A second bus: its `ρ'`, its grid sums `U'` and its weight's product so
    /// far.
    second: Option<SecondState<E>>,
    /// The challenges bound so far.
    bound: Vec<FieldElement<E>>,
    /// The message of the round under way, once asked for.
    pending: Option<Vec<FieldElement<E>>>,
    /// A round queued on the card by [`Self::prefetch`], its rows unread.
    queued: Option<usize>,
    started: std::time::Instant,
    times: FusedTimes,
}

/// [`FusedStepper`]'s second bus.
#[cfg(feature = "cuda")]
struct SecondState<E: IsField> {
    rho: Vec<FieldElement<E>>,
    u: [[FieldElement<E>; 2]; 2],
    e_rho: FieldElement<E>,
}

#[cfg(feature = "cuda")]
impl<'f, E> FusedStepper<'f, E>
where
    E: IsField + 'static,
    FieldElement<E>: Send + Sync,
{
    /// Sets the session up and runs the grid pass. A decline leaves the
    /// transcript untouched. The corner check can also reject the proof.
    pub(crate) fn new<B>(
        source: FusedSource<'f>,
        input: &FusedInput<'_, B, E>,
        lambdas: &[FieldElement<E>],
        claim: &FieldElement<E>,
        degree: usize,
        check_corners: bool,
    ) -> Result<Result<Self, crate::Error>, ZerocheckDecline>
    where
        B: IsField + math::field::traits::IsSubFieldOf<E> + 'static,
    {
        use crate::fused::BusColumn;
        use crate::gpu::{ext3_from_raw, ext3_raw};
        use math::field::extensions_goldilocks::Degree3GoldilocksExtensionField as Ext3;
        use std::time::Instant;

        let unsupported = || ZerocheckDecline::new("unsupported fused zerocheck", 0);

        // Three rules, or five with a second bus.
        let rules = if input.second.is_some() { 5 } else { 3 };
        if std::any::TypeId::of::<E>() != std::any::TypeId::of::<Ext3>() || lambdas.len() != rules {
            return Err(unsupported());
        }
        let started = Instant::now();
        let device = source.view();
        let rows = device.rows;
        let num_vars = rows.trailing_zeros() as usize;
        let constraints = input.constraints;
        let d = constraints.degree().max(1);
        if degree != d + 1 || input.r.len() != num_vars || input.claim_point.len() < num_vars {
            return Err(unsupported());
        }
        // Rounds 0 and 1 on the grid, then the later rounds down to where today's
        // hand the cube to the host.
        let there = num_vars.saturating_sub(crate::HOST_CUBE_COMPILED.trailing_zeros() as usize);
        if there < 2 || input.betas.len() != constraints.roots().len() {
            return Err(unsupported());
        }
        let lowered = lower_fused(constraints).ok_or_else(unsupported)?;
        // A long program's rounds stay on the card past today's crossover
        // ([`zc_device_tail`]), except under the cross-check, which
        // replays today's rounds and so stops where they do.
        let crossover = there;
        let steps = lowered.nodes.len() / 2;
        let there = match zc_device_tail() {
            Some((min_steps, cube)) if steps >= min_steps && !check_corners => num_vars
                .saturating_sub(cube.trailing_zeros() as usize)
                .max(crossover),
            _ => crossover,
        };
        let (interaction_point, rho) = input
            .claim_point
            .split_at(input.claim_point.len() - num_vars);
        let mut bus = BusColumn::new(
            input.interactions,
            interaction_point,
            &lambdas[1],
            &lambdas[2],
        )
        .map_err(|e| ZerocheckDecline::new(format!("bus column: {e}"), 0))?;
        if faults().bus
            && let Some((slot, a)) = bus.terms().first().cloned()
        {
            let mut terms = bus.terms().to_vec();
            terms[0] = (slot, a + FieldElement::<E>::one());
            bus = BusColumn::from_parts(bus.constant().clone(), terms);
        }
        // The second bus's column at `λ³, λ⁴`, and its `ρ'`.
        let second = match &input.second {
            None => None,
            Some(second) => {
                if second.claim_point.len() < num_vars {
                    return Err(unsupported());
                }
                let (point, rho) = second
                    .claim_point
                    .split_at(second.claim_point.len() - num_vars);
                let column = BusColumn::new(second.interactions, point, &lambdas[3], &lambdas[4])
                    .map_err(|e| ZerocheckDecline::new(format!("bus column: {e}"), 0))?;
                Some((column, rho.to_vec()))
            }
        };
        let keep = faults().keep_corners;
        let check_corners = check_corners && !keep;
        let points: Vec<(u32, u32)> = (0..=d as u32)
            .flat_map(|a| (0..=d as u32).map(move |b| (a, b)))
            .filter(|&(a, b)| check_corners || keep || !(a < 2 && b < 2))
            .collect();
        let walked: Vec<u32> = core::iter::once(0).chain(2..=d as u32).collect();
        let bus_rows = if second.is_some() { 5 } else { 3 };
        if points.len() > math_cuda::argue_fused::MAX_ROWS
            || walked.len() + bus_rows > math_cuda::argue_fused::MAX_ROWS
        {
            return Err(unsupported());
        }

        let raw = |v: &FieldElement<E>| ext3_raw(v);
        let flat = |vs: &[FieldElement<E>]| -> Option<Vec<u64>> {
            let mut out = Vec::with_capacity(vs.len() * 3);
            for v in vs {
                out.extend_from_slice(&raw(v)?);
            }
            Some(out)
        };
        let betas = flat(input.betas).ok_or_else(unsupported)?;
        let slots: Vec<u32> = bus
            .terms()
            .iter()
            .map(|(slot, _)| u32::try_from(*slot).ok())
            .collect::<Option<_>>()
            .ok_or_else(unsupported)?;
        if slots.iter().any(|&s| s as usize >= device.width)
            || constraints.reads().iter().any(|&s| s >= device.width)
        {
            return Err(unsupported());
        }
        let coeffs = flat(
            &bus.terms()
                .iter()
                .map(|(_, a)| a.clone())
                .collect::<Vec<_>>(),
        )
        .ok_or_else(unsupported)?;
        let r_tail = flat(&input.r[2..]).ok_or_else(unsupported)?;
        let rho_tail = flat(&rho[2..]).ok_or_else(unsupported)?;
        let program = lowered.device_program(&betas);
        let fused_bus = math_cuda::argue_fused::FusedBus {
            slots: &slots,
            coeffs: &coeffs,
            constant: raw(bus.constant()).ok_or_else(unsupported)?,
        };
        // The second bus's terms, as the first's.
        let second_parts = match &second {
            None => None,
            Some((column, rho2)) => {
                let slots: Vec<u32> = column
                    .terms()
                    .iter()
                    .map(|(slot, _)| u32::try_from(*slot).ok())
                    .collect::<Option<_>>()
                    .ok_or_else(unsupported)?;
                if slots.iter().any(|&s| s as usize >= device.width) {
                    return Err(unsupported());
                }
                let coeffs = flat(
                    &column
                        .terms()
                        .iter()
                        .map(|(_, a)| a.clone())
                        .collect::<Vec<_>>(),
                )
                .ok_or_else(unsupported)?;
                Some((
                    slots,
                    coeffs,
                    raw(column.constant()).ok_or_else(unsupported)?,
                    flat(&rho2[2..]).ok_or_else(unsupported)?,
                ))
            }
        };
        let second_bus = second_parts
            .as_ref()
            .map(
                |(slots, coeffs, constant, rho_tail)| math_cuda::argue_fused::SecondBus {
                    bus: math_cuda::argue_fused::FusedBus {
                        slots,
                        coeffs,
                        constant: *constant,
                    },
                    rho_tail,
                },
            );
        let mut asked = 0;
        let mut refused = None;
        let mut session = match math_cuda::argue_fused::FusedZerocheck::new(
            device,
            program,
            fused_bus,
            &r_tail,
            &rho_tail,
            points.len(),
            walked.len() + 2,
            second_bus,
            |be, bytes| {
                asked = bytes;
                reserve_zerocheck(|| be.try_reserve(bytes), || bytes <= be.vram_budget_bytes())
                    .map_err(|free| refused = Some(free))
                    .ok()
            },
        ) {
            Ok(Some(session)) => session,
            Ok(None) => {
                return Err(match refused {
                    Some(free) => ZerocheckDecline {
                        reason: "fused session reservation refused".into(),
                        asked,
                        free: Some(free),
                    },
                    None => unsupported(),
                });
            }
            Err(error) => {
                return Err(ZerocheckDecline::new(
                    format!("fused session setup: {error:?}"),
                    asked,
                ));
            }
        };
        // No round slot here: through one the zerocheck's launches ran slower
        // on whole blocks than with today's second launch, unlike the GKR
        // rounds'.

        let from = |limbs: &[u64]| ext3_from_raw::<E>(limbs);
        let side = d + 1;
        let grid_split = d == 3
            && !check_corners
            && !keep
            && zc_grid_split()
            && lower_fused_grid_split(constraints).is_some_and(|parts| {
                session
                    .prepare_grid_split(parts.each_ref().map(|p| p.device_program(&betas)))
                    .unwrap_or(false)
            });

        // ── rounds 0 and 1: one base-field pass ──
        let tick = Instant::now();
        let (t_sums, u_sums, violation) = session
            .grid(&points, keep, grid_split.then_some(&LOW_GRID_POINTS))
            .map_err(|error| ZerocheckDecline::new(format!("fused grid: {error:?}"), asked))?;
        if check_corners {
            if let Some(row) = violation
                && !faults().corners_inert
            {
                eprintln!(
                    "[argue] FUSED XCHECK: the trace breaks its constraints at row {row} (2^{num_vars} rows)"
                );
                return Ok(Err(crate::Error::ConstraintViolated { row: row as usize }));
            }
            CORNER_CHECKS.fetch_add(1, Ordering::Relaxed);
        }
        let t = if grid_split {
            let mut grid = std::array::from_fn(|_| std::array::from_fn(|_| FieldElement::zero()));
            for (&(a, b), limbs) in points.iter().zip(t_sums.chunks_exact(3)) {
                grid[a as usize][b as usize] = from(limbs);
            }
            let low = std::array::from_fn(|i| {
                let at = (points.len() + i) * 3;
                from(&t_sums[at..at + 3])
            });
            add_low_grid(&mut grid, &low);
            grid.into_iter().map(Vec::from).collect()
        } else {
            let mut grid = vec![vec![FieldElement::<E>::zero(); side]; side];
            for (&(a, b), limbs) in points.iter().zip(t_sums.chunks_exact(3)) {
                if a >= 2 || b >= 2 || keep {
                    grid[a as usize][b as usize] = from(limbs);
                }
            }
            grid
        };
        let u: [[FieldElement<E>; 2]; 2] = core::array::from_fn(|b| {
            core::array::from_fn(|c| from(&u_sums[(2 * b + c) * 3..(2 * b + c) * 3 + 3]))
        });
        let second = match second {
            None => None,
            Some((_, rho2)) => {
                let u2 = session
                    .second_grid()
                    .map_err(|error| {
                        ZerocheckDecline::new(format!("second fused grid: {error:?}"), asked)
                    })?
                    .ok_or_else(unsupported)?;
                Some(SecondState {
                    rho: rho2,
                    u: core::array::from_fn(|b| {
                        core::array::from_fn(|c| from(&u2[(2 * b + c) * 3..(2 * b + c) * 3 + 3]))
                    }),
                    e_rho: FieldElement::one(),
                })
            }
        };
        let times = FusedTimes {
            grid: tick.elapsed().as_secs_f64(),
            grid_split,
            roots: constraints.roots().len(),
            terms: slots.len(),
            slots: lowered.num_slots,
            ..FusedTimes::default()
        };
        Ok(Ok(Self {
            session,
            r: input.r.to_vec(),
            rho: rho.to_vec(),
            d,
            big: FieldElement::<E>::from(degree as u64),
            walked,
            there,
            crossover,
            steps,
            t,
            u,
            claim: claim.clone(),
            e_r: FieldElement::one(),
            e_rho: FieldElement::one(),
            second,
            bound: Vec::with_capacity(there),
            pending: None,
            queued: None,
            started,
            times,
        }))
    }

    /// Rounds the card runs before the host takes the cube.
    pub(crate) fn device_rounds(&self) -> usize {
        self.there
    }

    pub(crate) fn grid_was_split(&self) -> bool {
        self.times.grid_split
    }

    /// Variables this table has left to bind on the card.
    pub(crate) fn rounds_left(&self) -> usize {
        self.there - self.bound.len()
    }

    /// The claim the round under way carries in.
    pub(crate) fn claim(&self) -> &FieldElement<E> {
        &self.claim
    }

    /// Queues the round under way's launches on the card, not waited for, so
    /// the batched argue can queue every table's before it reads any back.
    /// Rounds 0 and 1 come from the grid pass and queue nothing.
    pub(crate) fn prefetch(&mut self) -> Result<(), crate::Error> {
        let j = self.bound.len();
        if j >= 2 && j < self.there && self.pending.is_none() && self.queued.is_none() {
            let rows = self.session.round_enqueue(&self.walked).map_err(|_| {
                crate::Error::DeviceFailed {
                    stage: "fused round",
                }
            })?;
            self.queued = Some(rows);
        }
        Ok(())
    }

    /// The round under way's message: its polynomial at `1..=d + 1`.
    pub(crate) fn message(&mut self) -> Result<Vec<FieldElement<E>>, crate::Error> {
        use crate::fused;
        use crate::gpu::ext3_from_raw;
        let failed = |stage| crate::Error::DeviceFailed { stage };
        let j = self.bound.len();
        if j >= self.there {
            return Err(failed("fused round"));
        }
        if let Some(g) = &self.pending {
            return Ok(g.clone());
        }
        let one = FieldElement::<E>::one();
        let d = self.d;
        let (r, rho) = (&self.r, &self.rho);
        let g = match j {
            0 => {
                let a0: Vec<FieldElement<E>> = (0..=d)
                    .map(|a| (&one - &r[1]) * &self.t[a][0] + &r[1] * &self.t[a][1])
                    .collect();
                let b0: [FieldElement<E>; 2] = core::array::from_fn(|b| {
                    (&one - &rho[1]) * &self.u[b][0] + &rho[1] * &self.u[b][1]
                });
                let g = fused::message(&r[0], &rho[0], &self.e_r, &self.e_rho, &a0, &b0, &self.big);
                match &self.second {
                    None => g,
                    Some(second) => {
                        let (rho2, u2) = (&second.rho, &second.u);
                        let b2: [FieldElement<E>; 2] = core::array::from_fn(|b| {
                            (&one - &rho2[1]) * &u2[b][0] + &rho2[1] * &u2[b][1]
                        });
                        fused::with_bus(g, &rho2[0], &second.e_rho, &b2)
                    }
                }
            }
            1 => {
                let s0 = &self.bound[0];
                let a1: Vec<FieldElement<E>> = (0..=d)
                    .map(|b| {
                        let column: Vec<FieldElement<E>> =
                            (0..=d).map(|a| self.t[a][b].clone()).collect();
                        crate::sumcheck::interpolate(&column, s0)
                    })
                    .collect();
                let b1: [FieldElement<E>; 2] =
                    core::array::from_fn(|c| (&one - s0) * &self.u[0][c] + s0 * &self.u[1][c]);
                let g = fused::message(&r[1], &rho[1], &self.e_r, &self.e_rho, &a1, &b1, &self.big);
                match &self.second {
                    None => g,
                    Some(second) => {
                        let u2 = &second.u;
                        let b2: [FieldElement<E>; 2] =
                            core::array::from_fn(|c| (&one - s0) * &u2[0][c] + s0 * &u2[1][c]);
                        fused::with_bus(g, &second.rho[1], &second.e_rho, &b2)
                    }
                }
            }
            _ => {
                let tick = std::time::Instant::now();
                let from = |limbs: &[u64]| ext3_from_raw::<E>(limbs);
                let rows = match self.queued.take() {
                    Some(rows) => rows,
                    None => self
                        .session
                        .round_enqueue(&self.walked)
                        .map_err(|_| failed("fused round"))?,
                };
                let sums = self
                    .session
                    .collect(rows)
                    .map_err(|_| failed("fused round"))?;
                let mut a = vec![FieldElement::<E>::zero(); d + 1];
                for (k, &node) in self.walked.iter().enumerate() {
                    a[node as usize] = from(&sums[k * 3..k * 3 + 3]);
                }
                let nb = self.walked.len();
                let b = [
                    from(&sums[nb * 3..nb * 3 + 3]),
                    from(&sums[nb * 3 + 3..nb * 3 + 6]),
                ];
                // The second bus's two sums follow the first's.
                let b2: Option<[FieldElement<E>; 2]> = self.second.as_ref().map(|_| {
                    [
                        from(&sums[(nb + 2) * 3..(nb + 2) * 3 + 3]),
                        from(&sums[(nb + 3) * 3..(nb + 3) * 3 + 3]),
                    ]
                });
                let (r_j, rho_j) = (&r[j], &rho[j]);
                let divisor = &self.e_r * r_j;
                let mut bus_part = fused::bus_sum(rho_j, &self.e_rho, &b);
                if let (Some(second), Some(b2)) = (&self.second, &b2) {
                    bus_part += fused::bus_sum(&second.rho[j], &second.e_rho, b2);
                }
                a[1] = match divisor.inv() {
                    Ok(inverse) => {
                        (&self.claim - &self.e_r * (&one - r_j) * &a[0] - bus_part) * inverse
                    }
                    // `E^r_j·r_j` vanished: the claim says nothing about `A(1)`, so
                    // it is walked.
                    Err(_) => match self.session.round(&[1]) {
                        Ok(direct) => from(&direct[0..3]),
                        Err(_) => return Err(failed("fused round")),
                    },
                };
                self.times.rounds += tick.elapsed().as_secs_f64();
                let g = fused::message(r_j, rho_j, &self.e_r, &self.e_rho, &a, &b, &self.big);
                match (&self.second, b2) {
                    (Some(second), Some(b2)) => {
                        fused::with_bus(g, &second.rho[j], &second.e_rho, &b2)
                    }
                    _ => g,
                }
            }
        };
        self.pending = Some(g.clone());
        Ok(g)
    }

    /// Binds the round under way to `s`: the claim carried on, the weights'
    /// prefix products, and the fold on the card.
    pub(crate) fn bind(&mut self, s: &FieldElement<E>) -> Result<(), crate::Error> {
        use crate::fused;
        use crate::gpu::ext3_raw;
        let failed = |stage| crate::Error::DeviceFailed { stage };
        let j = self.bound.len();
        let g = self.pending.take().ok_or(failed("fused bind"))?;
        let mut all = Vec::with_capacity(g.len() + 1);
        all.push(&self.claim - &g[0]);
        all.extend(g.iter().cloned());
        self.claim = crate::sumcheck::interpolate(&all, s);
        self.e_r = &self.e_r * fused::eq1(&self.r[j], s);
        self.e_rho = &self.e_rho * fused::eq1(&self.rho[j], s);
        if let Some(second) = self.second.as_mut() {
            second.e_rho = &second.e_rho * fused::eq1(&second.rho[j], s);
        }
        self.bound.push(s.clone());
        match j {
            0 => {}
            1 => {
                // ── both folds at once ──
                let tick = std::time::Instant::now();
                let (s0, s1) = (&self.bound[0], &self.bound[1]);
                let mut w = [0u64; 12];
                for a in 0..2u64 {
                    for b in 0..2u64 {
                        let wab = fused::eq1(s0, &FieldElement::<E>::from(a))
                            * fused::eq1(s1, &FieldElement::<E>::from(b));
                        let at = (2 * a + b) as usize * 3;
                        let limbs = ext3_raw(&wab).ok_or(failed("fused weights"))?;
                        w[at..at + 3].copy_from_slice(&limbs);
                    }
                }
                // Rounds 0 and 1's challenges: the rows' bus values are folded by
                // them (`argue_fused::bus_rows`).
                let mut challenges = [0u64; 6];
                challenges[..3].copy_from_slice(&ext3_raw(s0).ok_or(failed("fused weights"))?);
                challenges[3..].copy_from_slice(&ext3_raw(s1).ok_or(failed("fused weights"))?);
                self.session
                    .fold2(&w, &challenges)
                    .map_err(|_| failed("fused double fold"))?;
                self.times.fold2 = tick.elapsed().as_secs_f64();
            }
            _ => {
                let tick = std::time::Instant::now();
                let limbs = ext3_raw(s).ok_or(failed("fused challenge"))?;
                self.session
                    .fold(&limbs, j + 1 < self.there)
                    .map_err(|_| failed("fused fold"))?;
                self.times.rounds += tick.elapsed().as_secs_f64();
            }
        }
        Ok(())
    }

    /// What the host tail reads once the card's rounds are bound: the trace's
    /// factors as folded, then the two weights folded by the same challenges —
    /// `E^·` times `eq` of the rest.
    pub(crate) fn into_factors(
        mut self,
    ) -> Result<(Vec<crate::mle::Mle<E>>, FusedTimes), crate::Error> {
        use crate::gpu::ext3_from_raw;
        let failed = |stage| crate::Error::DeviceFailed { stage };
        if self.bound.len() != self.there {
            return Err(failed("fused factor values"));
        }
        let values = self
            .session
            .values()
            .map_err(|_| failed("fused factor values"))?;
        let mut factors: Vec<crate::mle::Mle<E>> = Vec::with_capacity(values.len() + 2);
        for factor in &values {
            factors.push(
                crate::mle::Mle::new(factor.chunks_exact(3).map(ext3_from_raw::<E>).collect())
                    .map_err(|_| failed("fused factor values"))?,
            );
        }
        let mut weights = vec![
            (&self.e_r, &self.r[self.there..]),
            (&self.e_rho, &self.rho[self.there..]),
        ];
        if let Some(second) = &self.second {
            weights.push((&second.e_rho, &second.rho[self.there..]));
        }
        for (scale, tail) in weights {
            let table: Vec<FieldElement<E>> = crate::eq::eq_evals(tail)
                .into_iter()
                .map(|v| v * scale)
                .collect();
            factors.push(crate::mle::Mle::new(table).map_err(|_| failed("fused weights"))?);
        }
        SESSIONS.fetch_add(1, Ordering::Relaxed);
        crate::gpu::note_device_sumcheck(self.there as u64);
        self.times.total = self.started.elapsed().as_secs_f64();
        let past = self.there > self.crossover;
        if past {
            DEVICE_TAILS.fetch_add(1, Ordering::Relaxed);
        }
        let long = usize::from(self.long());
        let rounds = self.times.rounds;
        note_tail(|tail| {
            tail.tables[long] += 1;
            tail.on_card[long] += u32::from(past);
            tail.device[long] += rounds;
        });
        Ok((factors, self.times))
    }

    /// Whether the program is one the device tail keeps on the card, knob on
    /// or off: at least [`DEVICE_TAIL_MIN_STEPS`] steps.
    pub(crate) fn long(&self) -> bool {
        self.steps >= DEVICE_TAIL_MIN_STEPS
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use math::field::goldilocks::GoldilocksField as Gl;

    type FB = FieldElement<Gl>;

    #[cfg(feature = "cuda")]
    #[test]
    fn a_zerocheck_retries_once_when_the_opening_releases_its_room() {
        let (done, release) = std::sync::mpsc::channel();
        with_opening_release(Some(release), || {
            assert_eq!(reserve_zerocheck(|| Ok(1), || true), Ok(1));
            assert_eq!(reserve_zerocheck::<()>(|| Err(10), || false), Err(10));
            let mut done = Some(done);
            let reserve = || {
                if let Some(done) = done.take() {
                    drop(done);
                    Err(10)
                } else {
                    Ok(20)
                }
            };
            assert_eq!(reserve_zerocheck(reserve, || true), Ok(20));
            assert_eq!(reserve_zerocheck::<()>(|| Err(10), || true), Err(10));
        });
    }

    /// `x·y + 3` and `x − y` as two roots, the second selected by factor 2.
    fn two_roots() -> Constraints<Gl> {
        use math::field::extensions_goldilocks::Degree3GoldilocksExtensionField as Ext3;
        let steps: Vec<Op<Ext3>> = vec![
            Op::Var(0),
            Op::Var(1),
            Op::Mul(0, 1),
            Op::Fixed(FieldElement::<Ext3>::from(3u64)),
            Op::Add(2, 3),
            Op::Sub(0, 1),
        ];
        Constraints::<Gl>::from_extension(&steps, &[4, 5], &[None, Some(2)], 2).unwrap()
    }

    /// Walks the lowered blob on the host the way `zc_grid01` does, in the base
    /// field, and returns `Σ β_i·sel_i·root_i` with `β = (1, 10)`.
    fn run(lowered: &LoweredFused, factors: &[u64]) -> u64 {
        let mut slots = vec![FB::zero(); lowered.num_slots];
        let betas = [FB::from(1u64), FB::from(10u64)];
        let mut acc = FB::zero();
        for node in lowered.nodes.chunks_exact(2) {
            let (op, a) = ((node[0] & 0xFFFF_FFFF) as u32, (node[0] >> 32) as u32);
            let (b, res) = ((node[1] & 0xFFFF_FFFF) as u32, (node[1] >> 32) as u32);
            let v = |s: u32| slots[s as usize];
            match op {
                0 => slots[res as usize] = FB::from(lowered.base_consts[a as usize]),
                1 => slots[res as usize] = FB::from(factors[a as usize]),
                2 => slots[res as usize] = v(a) + v(b),
                3 => slots[res as usize] = v(a) - v(b),
                4 => slots[res as usize] = v(a) * v(b),
                5 => slots[res as usize] = -v(a),
                6 => {
                    let sel = if b == op::NO_SELECTOR {
                        FB::one()
                    } else {
                        FB::from(factors[b as usize])
                    };
                    acc += betas[res as usize] * v(a) * sel;
                }
                _ => unreachable!(),
            }
        }
        acc.canonical_u64()
    }

    #[test]
    fn the_lowered_blob_sums_the_selected_roots() {
        let lowered = lower_fused(&two_roots()).unwrap();
        // 2 roots → 2 ACC nodes; 6 steps.
        assert_eq!(lowered.nodes.len(), 2 * 8);
        assert_eq!(lowered.base_consts, vec![3]);
        assert_eq!(lowered.ext_consts, vec![3, 0, 0]);
        for (x, y, s) in [(2u64, 5u64, 1u64), (7, 7, 0), (4, 9, 3)] {
            let want = (FB::from(x) * FB::from(y) + FB::from(3u64))
                + FB::from(10u64) * (FB::from(x) - FB::from(y)) * FB::from(s);
            assert_eq!(
                run(&lowered, &[x, y, s]),
                want.canonical_u64(),
                "x={x} y={y} s={s}"
            );
        }
    }

    /// The device tail's rule, pinned: its constants as frozen before they were
    /// measured, below the host's crossover cube.
    #[test]
    fn the_device_tail_takes_long_programs_down_to_a_cube_of_4() {
        assert_eq!((DEVICE_TAIL_MIN_STEPS, DEVICE_TAIL_CUBE), (1000, 4));
        const { assert!(DEVICE_TAIL_CUBE < crate::HOST_CUBE_COMPILED) };
    }

    #[test]
    fn a_slot_is_reused_once_its_last_reader_passed() {
        let lowered = lower_fused(&two_roots()).unwrap();
        assert!(lowered.num_slots <= 4, "{} slots", lowered.num_slots);
    }
}

#[cfg(test)]
mod grid_split_tests {
    use super::*;
    use crate::{batch, sumcheck};
    use crypto::fiat_shamir::{default_transcript::DefaultTranscript, is_transcript::IsTranscript};
    use math::field::{
        extensions_goldilocks::Degree3GoldilocksExtensionField as Ext,
        goldilocks::GoldilocksField as Gl,
    };

    type Base = FieldElement<Gl>;
    type ExtValue = FieldElement<Ext>;
    type Grid = [[ExtValue; 4]; 4];

    #[test]
    fn the_grid_split_defaults_on_and_only_zero_disables_it() {
        for value in [None, Some(""), Some("1"), Some("other")] {
            assert!(zc_grid_split_from_env(value));
        }
        for value in ["0", " \t0\n"] {
            assert!(!zc_grid_split_from_env(Some(value)));
        }
    }

    fn evaluate<F: IsField>(
        col: &[FieldElement<F>; 4],
        a: &FieldElement<F>,
        b: &FieldElement<F>,
    ) -> FieldElement<F> {
        let lo = &col[0] + b * (&col[1] - &col[0]);
        let hi = &col[2] + b * (&col[3] - &col[2]);
        &lo + a * (&hi - &lo)
    }

    #[test]
    fn split_grid_preserves_the_sumcheck_and_the_verifier_refuses_bad_grids() {
        // The same quadratic root is cubic once multiplied by its selector.
        let constraints = Constraints::<Gl>::from_extension(
            &[
                Op::Var(0),
                Op::Fixed(ExtValue::one()),
                Op::Sub(0, 1),
                Op::Mul(0, 2),
                Op::Fixed(ExtValue::from(2)),
                Op::Sub(0, 4),
                Op::Mul(3, 5),
                Op::Var(1),
                Op::Var(2),
                Op::Sub(7, 8),
            ],
            &[3, 6, 9, 3],
            &[None, None, Some(3), Some(3)],
            3,
        )
        .unwrap();
        let whole = lower_fused(&constraints).unwrap();
        let [low, high] = lower_fused_grid_split(&constraints).unwrap();
        // Mutant: move x(x-1)(x-2) into the quadratic program.
        let misplaced_low = lower_fused_part(&constraints, Some(&[0, 1, 2])).unwrap();
        let misplaced_high = lower_fused_part(&constraints, Some(&[3])).unwrap();
        let betas = [2, 3, 5, 7].map(ExtValue::from);
        let rules = [batch::Rule::new(3, |values| {
            let [x, y, z, sel] = values else {
                unreachable!()
            };
            let quadratic = x * (x - ExtValue::one());
            &quadratic * (&betas[0] + (x - ExtValue::from(2)) * &betas[1] + sel * &betas[3])
                + &betas[2] * sel * (y - z)
        })];
        for k in 0u64..8 {
            let parity = k & 1;
            let cols = [
                [parity, 1 - parity, 1 - parity, parity],
                [3 + k, 7 + 2 * k, 11 + 3 * k, 13 + 5 * k],
                [4 + k, 7 + 2 * k, 6 + 3 * k, 13 + 5 * k],
                [0, 1, 0, 1],
            ]
            .map(|col| col.map(Base::from));
            let eval = |program, a, b| {
                let values = cols
                    .each_ref()
                    .map(|col| evaluate(col, &Base::from(u64::from(a)), &Base::from(u64::from(b))));
                run_lowered_host(program, &betas, &values)
            };
            let grid = |program| -> Grid {
                std::array::from_fn(|a| std::array::from_fn(|b| eval(program, a as u32, b as u32)))
            };
            let low_at_five = |program| LOW_GRID_POINTS.map(|(a, b)| eval(program, a, b));
            let whole_grid = grid(&whole);
            assert!(
                whole_grid[..2]
                    .iter()
                    .all(|row| row[..2].iter().all(|x| x == &ExtValue::zero()))
            );
            let low_sums = low_at_five(&low);
            let mut split_grid = grid(&high);
            add_low_grid(&mut split_grid, &low_sums);
            assert_eq!(split_grid, whole_grid);
            let verify = |t: &Grid| {
                let mut transcript = DefaultTranscript::<Ext>::new(b"constraint grid");
                transcript.append_field_element(&ExtValue::zero());
                let _: ExtValue = transcript.sample_field_element();
                let first = (1..4).map(|a| &t[a][0] + &t[a][1]).collect::<Vec<_>>();
                for v in &first {
                    transcript.append_field_element(v);
                }
                let a: ExtValue = transcript.sample_field_element();
                let second = (1..4)
                    .map(|b| sumcheck::interpolate(&t.each_ref().map(|row| row[b]), &a))
                    .collect();
                let proof = sumcheck::SumcheckProof {
                    rounds: vec![
                        sumcheck::RoundProof { evaluations: first },
                        sumcheck::RoundProof {
                            evaluations: second,
                        },
                    ],
                };
                batch::verify(
                    &proof,
                    &rules,
                    &[ExtValue::zero()],
                    |point| {
                        Ok(cols
                            .each_ref()
                            .map(|col| {
                                let lifted = col.map(|x| x.to_extension::<Ext>());
                                evaluate(&lifted, &point[0], &point[1])
                            })
                            .to_vec())
                    },
                    2,
                    &mut DefaultTranscript::<Ext>::new(b"constraint grid"),
                )
            };
            assert!(verify(&split_grid).is_ok());

            // Mutant: weight the low (2, 0) evaluation by 2 instead of 3 at (3, 0).
            let mut bad_weight = split_grid;
            bad_weight[3][0] = &bad_weight[3][0] - &low_sums[0];
            assert_eq!(verify(&bad_weight), Err(crate::Error::BatchMismatch));

            let mut bad_degree = grid(&misplaced_high);
            add_low_grid(&mut bad_degree, &low_at_five(&misplaced_low));
            assert_eq!(verify(&bad_degree), Err(crate::Error::BatchMismatch));
        }
    }
}
