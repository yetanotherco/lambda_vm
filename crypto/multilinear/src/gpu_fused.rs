//! D-ARGUE stage 1 on the card (`thoughts/zf/gap2/fix2/D-ARGUE.md` §4.2): a
//! table's zerocheck rounds with the bus as one column (S1-2), the weights
//! pulled out (S1-4), rounds 0 and 1 from one base-field pass (S1-5) and
//! integer nodes (S1-1) — every message today's.
//!
//! [`fused`](crate::fused) is the host reference and this is its device twin,
//! round for round: the same parts (`T`, `U`, `A`, `B`), the same messages out
//! of them ([`fused::message`]). The kernels are `math_cuda::argue_fused`'s.
//!
//! On by default for every table whose constraint part runs in the base field
//! — or only the committed widths `LAMBDA_VM_ARGUE_FUSED_WIDTHS` lists;
//! `LAMBDA_VM_ARGUE_FUSED=0` is the opt-out, and it is today's rounds exactly.
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
    enum Item {
        Step(usize),
        Acc(usize),
    }
    let steps = constraints.steps();
    let roots = constraints.roots();
    let selectors = constraints.selectors();
    // Each root's ACC right after the step that makes it.
    let mut after: Vec<Vec<usize>> = vec![Vec::new(); steps.len()];
    for (i, &root) in roots.iter().enumerate() {
        after.get_mut(root as usize)?.push(i);
    }
    let mut items = Vec::with_capacity(steps.len() + roots.len());
    for (s, accs) in after.iter().enumerate() {
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

/// Whether the zerocheck runs the fused rounds: on by default,
/// `LAMBDA_VM_ARGUE_FUSED=0` the opt-out. Read once, with a banner.
pub fn argue_fused() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    forced(&FUSED_FORCED, || {
        *ON.get_or_init(|| {
            let on = fused_from(std::env::var("LAMBDA_VM_ARGUE_FUSED").ok().as_deref());
            eprintln!(
                "★ ARGUE FUSED: {}",
                if on {
                    "on (the default; LAMBDA_VM_ARGUE_FUSED=0 is today's rounds)"
                } else {
                    "off (LAMBDA_VM_ARGUE_FUSED=0)"
                }
            );
            on
        })
    })
}

/// The knob's reading of its variable: only `0` turns it off.
fn fused_from(value: Option<&str>) -> bool {
    value != Some("0")
}

#[doc(hidden)]
pub fn force_argue_fused(on: Option<bool>) {
    store_forced(&FUSED_FORCED, on);
}

/// Whether a table of `width` committed columns is one the fused rounds take:
/// every table, or those `LAMBDA_VM_ARGUE_FUSED_WIDTHS` (comma-separated) names.
pub fn fused_takes_width(width: usize) -> bool {
    static WIDTHS: std::sync::OnceLock<Option<Vec<usize>>> = std::sync::OnceLock::new();
    WIDTHS
        .get_or_init(|| {
            std::env::var("LAMBDA_VM_ARGUE_FUSED_WIDTHS")
                .ok()
                .map(|list| {
                    list.split(',')
                        .filter_map(|w| w.trim().parse().ok())
                        .collect()
                })
        })
        .as_ref()
        .is_none_or(|widths| widths.contains(&width))
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

/// Whether every fused table prints its timings (`LAMBDA_VM_ARGUE_FUSED_LOG=1`);
/// the cross-check prints them regardless. Read once.
pub fn argue_fused_log() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| env_is("LAMBDA_VM_ARGUE_FUSED_LOG"))
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
    for c in [&SESSIONS, &DECLINES, &XCHECKS, &CORNER_CHECKS] {
        c.store(0, Ordering::Relaxed);
    }
}

/// One fused table's timings, for its log line.
#[derive(Clone, Copy, Debug, Default)]
pub struct FusedTimes {
    pub grid: f64,
    pub fold2: f64,
    pub rounds: f64,
    pub total: f64,
    /// The table's signature on the log line (a committed table has no name
    /// here): its roots, its bus column's terms, the constraint walk's slots.
    pub roots: usize,
    pub terms: usize,
    pub slots: usize,
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
/// `None` is a decline, always before the first message; `Some(Err)` after.
///
/// The rounds are [`FusedStepper`]'s, driven here one after the other; the
/// batched argue drives the same stepper in lockstep with other tables'.
#[cfg(feature = "cuda")]
#[allow(clippy::too_many_arguments)]
pub(crate) fn prove_fused<B, E>(
    resident: &crate::gpu::DeviceFactors,
    input: &FusedInput<'_, B, E>,
    lambdas: &[FieldElement<E>],
    claim: &FieldElement<E>,
    degree: usize,
    check_corners: bool,
    mut challenge: impl FnMut(&[FieldElement<E>]) -> FieldElement<E>,
) -> Option<Result<FusedRounds<E>, crate::Error>>
where
    B: IsField + math::field::traits::IsSubFieldOf<E> + 'static,
    E: IsField + 'static,
    FieldElement<E>: Send + Sync,
{
    let mut stepper =
        match FusedStepper::new(resident, input, lambdas, claim, degree, check_corners)? {
            Ok(stepper) => stepper,
            Err(error) => return Some(Err(error)),
        };
    let there = stepper.device_rounds();
    let mut rounds = Vec::with_capacity(there);
    let mut point = Vec::with_capacity(there);
    for _ in 0..there {
        let g = match stepper.message() {
            Ok(g) => g,
            Err(error) => return Some(Err(error)),
        };
        let s = challenge(&g);
        if let Err(error) = stepper.bind(&s) {
            return Some(Err(error));
        }
        rounds.push(crate::sumcheck::RoundProof { evaluations: g });
        point.push(s);
    }
    let (factors, times) = match stepper.into_factors() {
        Ok(done) => done,
        Err(error) => return Some(Err(error)),
    };
    Some(Ok((rounds, point, factors, times)))
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
    t: Vec<Vec<FieldElement<E>>>,
    u: [[FieldElement<E>; 2]; 2],
    claim: FieldElement<E>,
    e_r: FieldElement<E>,
    e_rho: FieldElement<E>,
    /// The challenges bound so far.
    bound: Vec<FieldElement<E>>,
    /// The message of the round under way, once asked for.
    pending: Option<Vec<FieldElement<E>>>,
    started: std::time::Instant,
    times: FusedTimes,
}

#[cfg(feature = "cuda")]
impl<'f, E> FusedStepper<'f, E>
where
    E: IsField + 'static,
    FieldElement<E>: Send + Sync,
{
    /// Sets the session up and runs the grid pass. `None` is a decline, before
    /// anything is sent; `Some(Err)` a trace the corner check refuses.
    pub(crate) fn new<B>(
        resident: &'f crate::gpu::DeviceFactors,
        input: &FusedInput<'_, B, E>,
        lambdas: &[FieldElement<E>],
        claim: &FieldElement<E>,
        degree: usize,
        check_corners: bool,
    ) -> Option<Result<Self, crate::Error>>
    where
        B: IsField + math::field::traits::IsSubFieldOf<E> + 'static,
    {
        use crate::fused::BusColumn;
        use crate::gpu::{ext3_from_raw, ext3_raw};
        use math::field::extensions_goldilocks::Degree3GoldilocksExtensionField as Ext3;
        use std::time::Instant;

        if std::any::TypeId::of::<E>() != std::any::TypeId::of::<Ext3>() || lambdas.len() != 3 {
            return None;
        }
        let started = Instant::now();
        let device = resident.inner();
        let rows = device.len();
        let num_vars = rows.trailing_zeros() as usize;
        let constraints = input.constraints;
        let d = constraints.degree().max(1);
        if degree != d + 1 || input.r.len() != num_vars || input.claim_point.len() < num_vars {
            return None;
        }
        // Rounds 0 and 1 on the grid, then the later rounds down to where today's
        // hand the cube to the host.
        let there = num_vars.saturating_sub(crate::HOST_CUBE_COMPILED.trailing_zeros() as usize);
        if there < 2 || input.betas.len() != constraints.roots().len() {
            return None;
        }
        let lowered = lower_fused(constraints)?;
        let (interaction_point, rho) = input
            .claim_point
            .split_at(input.claim_point.len() - num_vars);
        let mut bus = BusColumn::new(
            input.interactions,
            interaction_point,
            &lambdas[1],
            &lambdas[2],
        )
        .ok()?;
        if faults().bus
            && let Some((slot, a)) = bus.terms().first().cloned()
        {
            let mut terms = bus.terms().to_vec();
            terms[0] = (slot, a + FieldElement::<E>::one());
            bus = BusColumn::from_parts(bus.constant().clone(), terms);
        }
        let keep = faults().keep_corners;
        let check_corners = check_corners && !keep;
        let points: Vec<(u32, u32)> = (0..=d as u32)
            .flat_map(|a| (0..=d as u32).map(move |b| (a, b)))
            .filter(|&(a, b)| check_corners || keep || !(a < 2 && b < 2))
            .collect();
        let walked: Vec<u32> = core::iter::once(0).chain(2..=d as u32).collect();
        if points.len() > math_cuda::argue_fused::MAX_ROWS
            || walked.len() + 3 > math_cuda::argue_fused::MAX_ROWS
        {
            return None;
        }

        let raw = |v: &FieldElement<E>| ext3_raw(v);
        let flat = |vs: &[FieldElement<E>]| -> Option<Vec<u64>> {
            let mut out = Vec::with_capacity(vs.len() * 3);
            for v in vs {
                out.extend_from_slice(&raw(v)?);
            }
            Some(out)
        };
        let betas = flat(input.betas)?;
        let slots: Vec<u32> = bus
            .terms()
            .iter()
            .map(|(slot, _)| u32::try_from(*slot).ok())
            .collect::<Option<_>>()?;
        if slots.iter().any(|&s| s as usize >= device.width())
            || constraints.reads().iter().any(|&s| s >= device.width())
        {
            return None;
        }
        let coeffs = flat(
            &bus.terms()
                .iter()
                .map(|(_, a)| a.clone())
                .collect::<Vec<_>>(),
        )?;
        let r_tail = flat(&input.r[2..])?;
        let rho_tail = flat(&rho[2..])?;
        let program = math_cuda::argue_fused::FusedProgram {
            nodes: &lowered.nodes,
            base_consts: &lowered.base_consts,
            ext_consts: &lowered.ext_consts,
            betas: &betas,
            num_slots: lowered.num_slots,
        };
        let fused_bus = math_cuda::argue_fused::FusedBus {
            slots: &slots,
            coeffs: &coeffs,
            constant: raw(bus.constant())?,
        };
        let mut session = match math_cuda::argue_fused::FusedZerocheck::new(
            device,
            program,
            fused_bus,
            &r_tail,
            &rho_tail,
            points.len(),
            walked.len() + 2,
        ) {
            Ok(Some(session)) => session,
            _ => return None,
        };

        let from = |limbs: &[u64]| ext3_from_raw::<E>(limbs);
        let side = d + 1;

        // ── rounds 0 and 1: one base-field pass ──
        let tick = Instant::now();
        let Ok((t_sums, u_sums, violation)) = session.grid(&points, keep) else {
            return None;
        };
        if check_corners {
            if let Some(row) = violation
                && !faults().corners_inert
            {
                eprintln!(
                    "[argue] FUSED XCHECK: the trace breaks its constraints at row {row} (2^{num_vars} rows)"
                );
                return Some(Err(crate::Error::ConstraintViolated { row: row as usize }));
            }
            CORNER_CHECKS.fetch_add(1, Ordering::Relaxed);
        }
        let mut t = vec![vec![FieldElement::<E>::zero(); side]; side];
        for (k, &(a, b)) in points.iter().enumerate() {
            if a < 2 && b < 2 && !keep {
                continue;
            }
            t[a as usize][b as usize] = from(&t_sums[k * 3..k * 3 + 3]);
        }
        let u: [[FieldElement<E>; 2]; 2] = core::array::from_fn(|b| {
            core::array::from_fn(|c| from(&u_sums[(2 * b + c) * 3..(2 * b + c) * 3 + 3]))
        });
        let times = FusedTimes {
            grid: tick.elapsed().as_secs_f64(),
            roots: constraints.roots().len(),
            terms: slots.len(),
            slots: lowered.num_slots,
            ..FusedTimes::default()
        };
        Some(Ok(Self {
            session,
            r: input.r.to_vec(),
            rho: rho.to_vec(),
            d,
            big: FieldElement::<E>::from(degree as u64),
            walked,
            there,
            t,
            u,
            claim: claim.clone(),
            e_r: FieldElement::one(),
            e_rho: FieldElement::one(),
            bound: Vec::with_capacity(there),
            pending: None,
            started,
            times,
        }))
    }

    /// Rounds the card runs before the host takes the cube.
    pub(crate) fn device_rounds(&self) -> usize {
        self.there
    }

    /// Variables this table has left to bind on the card.
    pub(crate) fn rounds_left(&self) -> usize {
        self.there - self.bound.len()
    }

    /// The claim the round under way carries in.
    pub(crate) fn claim(&self) -> &FieldElement<E> {
        &self.claim
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
                fused::message(&r[0], &rho[0], &self.e_r, &self.e_rho, &a0, &b0, &self.big)
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
                fused::message(&r[1], &rho[1], &self.e_r, &self.e_rho, &a1, &b1, &self.big)
            }
            _ => {
                let tick = std::time::Instant::now();
                let from = |limbs: &[u64]| ext3_from_raw::<E>(limbs);
                let sums = self
                    .session
                    .round(&self.walked)
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
                let (r_j, rho_j) = (&r[j], &rho[j]);
                let divisor = &self.e_r * r_j;
                let bus_part = &self.e_rho * ((&one - rho_j) * &b[0] + rho_j * &b[1]);
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
                fused::message(r_j, rho_j, &self.e_r, &self.e_rho, &a, &b, &self.big)
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
                self.session
                    .fold2(&w)
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
        for (scale, tail) in [
            (&self.e_r, &self.r[self.there..]),
            (&self.e_rho, &self.rho[self.there..]),
        ] {
            let table: Vec<FieldElement<E>> = crate::eq::eq_evals(tail)
                .into_iter()
                .map(|v| v * scale)
                .collect();
            factors.push(crate::mle::Mle::new(table).map_err(|_| failed("fused weights"))?);
        }
        SESSIONS.fetch_add(1, Ordering::Relaxed);
        crate::gpu::note_device_sumcheck(self.there as u64);
        self.times.total = self.started.elapsed().as_secs_f64();
        Ok((factors, self.times))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use math::field::goldilocks::GoldilocksField as Gl;

    type FB = FieldElement<Gl>;

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

    /// The default, pinned: on unless the variable says `0`.
    #[test]
    fn the_fused_rounds_are_on_unless_the_knob_says_0() {
        assert!(fused_from(None));
        assert!(fused_from(Some("1")));
        assert!(fused_from(Some("")));
        assert!(!fused_from(Some("0")));
    }

    #[test]
    fn a_slot_is_reused_once_its_last_reader_passed() {
        let lowered = lower_fused(&two_roots()).unwrap();
        assert!(lowered.num_slots <= 4, "{} slots", lowered.num_slots);
    }
}
