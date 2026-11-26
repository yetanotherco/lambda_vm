//! ★ The block's recursion as LEVELS: each level one W-LFM proof of one LFM
//! program, proved by the block's own prover ([`super::wlfm`]).
//!
//! - **Level 1** verifies the whole base block in one program
//!   ([`WhirBlockPlan::closed_block_program`]): the leaf walk over every group,
//!   the bus closed in-program, the top's words published.
//! - **Level k + 1** verifies level k's proof in-guest (the W-leg,
//!   [`super::wlfm_leg`]) and republishes its words in order, so every level
//!   publishes the top's words and [`top_claims`] reads the last one's.
//! - The chain's length is a verifier constant, [`LevelStop`]: by default
//!   [`LevelStop::DEFAULT`], two levels (the base verified, then that proof
//!   verified once: a file of a few megabytes, the size the uncompressed block
//!   ships today); deeper chains stop at a group count instead, for the
//!   compression modes. The level count follows from the shapes alone.
//!
//! There are no leaves and no nodes: the unit of proving (a group of the
//! level's block) is apart from the unit of verifying (a program).
//!
//! # Never from a proof
//!
//! Every level's program and artifacts come from the plan (the trusted ELF and
//! the statement) and the level below's artifacts ([`derive_levels_last`]): a
//! verifier derives them all, with every prepared stack committed on the host,
//! and checks only the last proof ([`verify_levels`]); each level below is
//! checked by the one above it, in-guest. A proof is read only to fill the next
//! level's arena — the witness.
//!
//! # The format is a constant
//!
//! Every level proves at [`LEVEL_RATE`] under today's levers and the family the
//! caller names, a verifier constant: [`WlfmFamily::P1`] over the block
//! family's Poseidon1 base, or [`WlfmFamily::B3`] over the 7-round BLAKE3 base
//! (`BaseFormat::B3_WHIR`), whose levels hash with the lean tables' rows. The
//! plan's hash, the base's and the levels' are one ([`level_plan`],
//! [`check_base`]) — one hash end to end.

use super::LfmWord;
use super::compiler::LfmProgram;
use super::edsl::WrapHash;
use super::last_level::LastLevel;
use super::merkle_cap::CapMux;
use super::whir_block::{WhirBlockPlan, block_leaf_arena, top_claims};
use super::wlfm::{
    StackPlace, WlfmBuild, WlfmFamily, WlfmLevers, WlfmPrebuild, WlfmProof, WlfmProving, WlfmRate,
    prepare_wlfm_rounds, prove_wlfm_traces_as, verify_wlfm,
};
use super::wlfm_leg::{
    WlfmChild, wlfm_leg_arena, wlfm_leg_program, wlfm_leg_program_marked, wlfm_wrap_hash,
};
use crate::block_whir::BlockWhirProof;
use crate::hash_pin::{BlockFamily, ProofFamily};
use multilinear::whir_chain::GroupFold;
use stark::proof::options::BaseFormat;
use std::sync::mpsc;

/// The rate every level proves at: the base's, blowup 2.
pub const LEVEL_RATE: WlfmRate = WlfmRate::Two;

/// `LAMBDA_VM_LEVELS_PREFIX_CHECK=1`: the levels' prover checks each table's
/// preprocessed prefix against the program's before it commits, in a release
/// build too ([`prefix_check`]).
pub const PREFIX_CHECK_ENV: &str = "LAMBDA_VM_LEVELS_PREFIX_CHECK";

/// Whether the levels' prover runs its prefix self-check: in a debug build, or
/// when [`PREFIX_CHECK_ENV`] says so; not in a release prove. It is the
/// prover checking itself and nothing more. The verifier binds every table's
/// preprocessed prefix through the prepared openings, so a prefix the check
/// would have caught only makes a proof the verifier rejects, never one it
/// accepts. Off, a release prove saves the check's pass over every prefix.
pub fn prefix_check() -> bool {
    if cfg!(debug_assertions) {
        return true;
    }
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var(PREFIX_CHECK_ENV).is_ok_and(|v| v.trim() == "1"))
}

/// Where a chain stops: a verifier constant, named by both sides.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum LevelStop {
    /// Exactly this many levels (at least one).
    Levels(usize),
    /// At the first level whose proof has at most this many groups.
    Groups(usize),
}

impl LevelStop {
    /// Two levels: level 1 over the base, level 2 over level 1.
    pub const DEFAULT: Self = LevelStop::Levels(2);
}

/// The most levels a chain may take: a [`LevelStop::Groups`] chain that does
/// not shrink to its count by then is refused rather than proved without end.
pub const MAX_LEVELS: usize = 8;

/// One level: its program and its build (the artifacts, and for a prover the
/// prepared stacks' openings).
pub struct Level {
    pub program: LfmProgram,
    pub build: WlfmBuild,
}

/// The levers every level proves under.
pub fn level_levers(family: WlfmFamily) -> WlfmLevers {
    WlfmLevers::with_family(family)
}

/// The levers level `index` (from 1) proves under: every level's, or the
/// last level's own ([`LastLevel::levers`]) when it is the last.
pub fn levers_for(family: WlfmFamily, last: LastLevel, is_last: bool) -> WlfmLevers {
    if is_last {
        last.levers(family)
    } else {
        level_levers(family)
    }
}

/// Whether level `index` (from 1) is the chain's last under `stop`, its
/// group count `groups`: the derivation's and the driver's one rule (a
/// [`LevelStop::Groups`] chain stops at [`MAX_LEVELS`] at the latest).
pub fn is_last_level(stop: LevelStop, index: usize, groups: usize) -> bool {
    match stop {
        LevelStop::Levels(n) => index >= n.clamp(1, MAX_LEVELS),
        LevelStop::Groups(g) => groups <= g || index >= MAX_LEVELS,
    }
}

/// Only a format proved end to end is taken ([`LastLevel::is_tested`]:
/// every lever off, every lever on, or one lever alone).
///
/// The last level's own format is the BLAKE3 levels' alone: every lever of
/// it is tested there (the lean tables, the binary trees); under
/// [`WlfmFamily::P1`] (4-ary trees) any lever on ([`LastLevel::OFF`] is
/// the only format there) is refused rather than proved untested.
pub fn check_last_level_family(last: LastLevel, family: WlfmFamily) -> Result<(), String> {
    if !last.is_tested() {
        return Err(format!(
            "the last level's format ({last}) has not been proved end to end: every lever off, \
             every lever on or one lever alone"
        ));
    }
    if family == WlfmFamily::P1 && !last.is_off() {
        return Err(format!(
            "the last level's own format ({last}) is the BLAKE3 levels' alone, not {family:?}'s"
        ));
    }
    Ok(())
}

/// Whether level `index` is known to be the last before it is prepared:
/// under [`LevelStop::Levels`] by its index; under [`LevelStop::Groups`] not
/// until its group count is known ([`prepare_as_level`]).
fn known_last(stop: LevelStop, index: usize) -> bool {
    match stop {
        LevelStop::Levels(_) => is_last_level(stop, index, usize::MAX),
        LevelStop::Groups(_) => index >= MAX_LEVELS,
    }
}

/// Level `index`'s prebuild by `prepare` under the levers it proves at: as
/// the last level when it is known to be ([`known_last`]), else every
/// level's, prepared again under the last level's when its group count makes
/// it the last ([`LevelStop::Groups`]; a group count does not depend on the
/// levers, `block_groups` reads the shapes and the stack alone). Under
/// [`LastLevel::OFF`] every level's levers are one, and nothing is
/// prepared twice.
pub(crate) fn prepare_as_level(
    stop: LevelStop,
    index: usize,
    last: LastLevel,
    family: WlfmFamily,
    prepare: impl Fn(WlfmLevers) -> Result<WlfmPrebuild, String>,
) -> Result<WlfmPrebuild, String> {
    check_last_level_family(last, family)?;
    let pre = prepare(levers_for(family, last, known_last(stop, index)))?;
    let a = pre.artifacts();
    let want = levers_for(family, last, is_last_level(stop, index, a.sizes.len()));
    if a.levers() == want {
        Ok(pre)
    } else {
        prepare(want)
    }
}

/// The levels' family against the hash the plan's programs replay: one hash
/// end to end, so a plan hashing otherwise than `family` is refused.
pub(crate) fn check_hash(wrap: WrapHash, family: WlfmFamily) -> Result<(), String> {
    if wlfm_wrap_hash(family) != wrap {
        return Err(format!(
            "the levels prove under {family:?} and the plan's programs hash under {wrap:?}: one \
             hash end to end"
        ));
    }
    Ok(())
}

fn check_family(plan: &WhirBlockPlan, family: WlfmFamily) -> Result<(), String> {
    check_hash(plan.wrap_hash(), family)
}

/// The base a family's levels verify: the block family's Poseidon1 base
/// under [`WlfmFamily::P1`], the 7-round BLAKE3 base under
/// [`WlfmFamily::B3`]. Any other pairing is refused before anything is
/// proved or derived.
pub fn check_base(base: &BaseFormat, family: WlfmFamily) -> Result<(), String> {
    let ok = match family {
        WlfmFamily::P1 => crate::block_whir::checked_base(base).is_ok(),
        WlfmFamily::B3 => *base == BaseFormat::B3_WHIR,
    };
    if !ok {
        return Err(format!(
            "{family:?} levels verify their own hash's base, not {:?} (4-ary cap {:?})",
            base.hash, base.arity4_cap
        ));
    }
    Ok(())
}

/// ★ Level 1's plan: one leaf over every group, under `family`'s derivation
/// (the block family's for [`WlfmFamily::P1`],
/// [`WhirBlockPlan::derive_blake3`] for [`WlfmFamily::B3`]). The prover hands
/// its prepared roots in; a verifier passes `None` and commits them on the
/// host. A prover's plan skips the chip rows' measuring under
/// [`PLAN_LEAN_ENV`] ([`level_plan_measured`]).
pub fn level_plan(
    elf: &[u8],
    proof_options: &crate::ProofOptions,
    format: &crate::block_whir::BlockFormat,
    statement: crate::block_whir::BlockStatement<'_>,
    family: WlfmFamily,
    prepared_roots: Option<&[(usize, Vec<multilinear::whir_commit::Commitment>)]>,
) -> Result<WhirBlockPlan, String> {
    let measure = !(plan_lean() && prepared_roots.is_some());
    level_plan_measured(
        elf,
        proof_options,
        format,
        statement,
        family,
        prepared_roots,
        measure,
    )
}

/// [`level_plan`] with the measuring named: `measure` false skips each
/// group's chip rows measured by emission, which a level's fixed leaf count
/// never reads ([`WhirBlockPlan::chip_loads`] is then empty). The same
/// programs either way.
pub(crate) fn level_plan_measured(
    elf: &[u8],
    proof_options: &crate::ProofOptions,
    format: &crate::block_whir::BlockFormat,
    statement: crate::block_whir::BlockStatement<'_>,
    family: WlfmFamily,
    prepared_roots: Option<&[(usize, Vec<multilinear::whir_commit::Commitment>)]>,
    measure: bool,
) -> Result<WhirBlockPlan, String> {
    level_plan_with(
        elf,
        proof_options,
        format,
        statement,
        family,
        prepared_roots,
        super::whir_block::share_inverse(),
        None,
        measure,
    )
}

/// [`level_plan`] with the leaves' share form named (`share_inverse`,
/// `whir_block::emit_share`) rather than the process's knob, and the
/// programs' cap check named (`cap_mux`; `None` reads the process's knob)
/// before the derivation emits anything; `measure` as
/// [`level_plan_measured`]'s (a verifier always measures).
#[allow(clippy::too_many_arguments)]
fn level_plan_with(
    elf: &[u8],
    proof_options: &crate::ProofOptions,
    format: &crate::block_whir::BlockFormat,
    statement: crate::block_whir::BlockStatement<'_>,
    family: WlfmFamily,
    prepared_roots: Option<&[(usize, Vec<multilinear::whir_commit::Commitment>)]>,
    share_inverse: bool,
    cap_mux: Option<CapMux>,
    measure: bool,
) -> Result<WhirBlockPlan, String> {
    crate::base_only::refuse_recursion()?;
    check_base(&format.zf.base, family)?;
    let plan = match family {
        WlfmFamily::P1 => WhirBlockPlan::derive_full(
            elf,
            proof_options,
            format,
            statement,
            Some(1),
            super::whir_block::BLOCK_FAN_IN,
            prepared_roots,
            super::whir_block::LEAF_P1_CAP,
            measure,
            share_inverse,
            cap_mux,
        )?,
        WlfmFamily::B3 => WhirBlockPlan::derive_blake3_with_share(
            elf,
            proof_options,
            format,
            statement,
            1,
            prepared_roots,
            measure,
            share_inverse,
            cap_mux,
        )?,
    };
    check_family(&plan, family)?;
    Ok(plan)
}

/// Card bytes a level's prepared stacks take committed there, per prepared
/// cell: their codewords at blowup 2 and their trees (a 2^20-compression level
/// probe: ≈ 250 M prepared cells, 4.98 GiB promised at its prove's start).
const STACK_CARD_BYTES_PER_CELL: u64 = 20;

/// The card room a group's commit needs above what the card already holds
/// (the same probe: 8.05 → 23.45 GiB through its first group's commit).
const GROUP_COMMIT_ROOM: u64 = 19 << 30;

/// `LAMBDA_VM_LEVELS_STACKS`: where the levels' prover commits each level's
/// prepared stacks. `auto` (unset or empty, the default): on the card, held
/// through the prove where the card holds them beside a group's commit, else
/// committed and let go at once and revived at each group's opening
/// ([`StackPlace::CardRetired`]); `held`, `retired`: that, on the card,
/// whatever the room; `host`: encoded and hashed on the host (a measurement's
/// control). Without a card, the host whatever it says. The same roots in
/// every place, so the same proofs: a prover's choice, never a format.
pub const STACKS_ENV: &str = "LAMBDA_VM_LEVELS_STACKS";

/// What [`STACKS_ENV`] says: `None` for `auto`; nonsense is an error.
pub fn parse_stacks(value: Option<&str>) -> Result<Option<StackPlace>, String> {
    match value.map(str::trim) {
        None | Some("" | "auto") => Ok(None),
        Some("held") => Ok(Some(StackPlace::Card)),
        Some("retired") => Ok(Some(StackPlace::CardRetired)),
        Some("host") => Ok(Some(StackPlace::Host)),
        Some(v) => Err(format!(
            "{STACKS_ENV} must be auto, held, retired or host, got `{v}`"
        )),
    }
}

/// Where level `level`'s `cells` prepared cells are committed: `forced` when
/// given, else on the card, held when the card's ledger budget, read now,
/// holds them beside a group's commit and retired when it does not; the host
/// only when there is no card. Printed.
pub(crate) fn stack_place(cells: u64, level: usize, forced: Option<StackPlace>) -> StackPlace {
    let need = cells.saturating_mul(STACK_CARD_BYTES_PER_CELL);
    let budget = multilinear::gpu::reserve_budget().saturating_sub(levels_pad_held());
    let place = if budget == 0 {
        StackPlace::Host
    } else if let Some(place) = forced {
        place
    } else if need > budget.saturating_sub(GROUP_COMMIT_ROOM) {
        StackPlace::CardRetired
    } else {
        StackPlace::Card
    };
    let gib = |b: u64| b as f64 / (1u64 << 30) as f64;
    eprintln!(
        "LEVELS STACKS level {level}: {cells} prepared cells · ≈ {:.2} GiB on the card · budget \
         {:.2} GiB · committed on the {}",
        gib(need),
        gib(budget),
        match place {
            StackPlace::Host => "host",
            StackPlace::Card => "card, held",
            StackPlace::CardRetired => "card, retired",
        }
    );
    place
}

/// The place [`STACKS_ENV`] forces, read once a process (a test's
/// [`set_stacks_for_test`] first).
fn stacks_forced() -> Result<Option<StackPlace>, String> {
    #[cfg(test)]
    if let Some(place) = *STACKS_FOR_TEST.lock().unwrap_or_else(|e| e.into_inner()) {
        return Ok(place);
    }
    static FORCED: std::sync::OnceLock<Result<Option<StackPlace>, String>> =
        std::sync::OnceLock::new();
    FORCED
        .get_or_init(|| parse_stacks(std::env::var(STACKS_ENV).ok().as_deref()))
        .clone()
}

/// A test's placement, for every thread of the process (the driver commits
/// on threads of its own): `Some(p)` as [`STACKS_ENV`] would say `p`.
#[cfg(test)]
static STACKS_FOR_TEST: std::sync::Mutex<Option<Option<StackPlace>>> = std::sync::Mutex::new(None);

/// Test only: the real-block harness's memory-keeping probe
/// (`RTA_KEEP_MEMORY=1`). When set, [`prove_block_levels`] keeps what the
/// process already holds before its clock starts: it skips the allocator's
/// purge and the card pool's trim, so a second block reuses the first's pages
/// and the pool's blocks. A production build has no such switch.
#[cfg(test)]
pub(crate) static KEEP_MEMORY_FOR_TEST: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(false);

#[cfg(test)]
fn keep_memory() -> bool {
    KEEP_MEMORY_FOR_TEST.load(std::sync::atomic::Ordering::Relaxed)
}

#[cfg(not(test))]
const fn keep_memory() -> bool {
    false
}

/// Sets [`STACKS_FOR_TEST`] until the guard drops.
#[cfg(test)]
pub(crate) fn set_stacks_for_test(place: Option<StackPlace>) -> StacksForTest {
    let before = STACKS_FOR_TEST
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .replace(place);
    StacksForTest(before)
}

#[cfg(test)]
pub(crate) struct StacksForTest(Option<Option<StackPlace>>);

#[cfg(test)]
impl Drop for StacksForTest {
    fn drop(&mut self) {
        *STACKS_FOR_TEST.lock().unwrap_or_else(|e| e.into_inner()) = self.0;
    }
}

/// `LAMBDA_VM_LEVELS_PROGRAM_FAST`: level 1's program is ready sooner, so
/// more of its execution runs beside the base's phase B. The driver compiles
/// it in parallel ([`super::compiler::compile_parallel`], the same program
/// row for row) and does not validate it before executing: the admission
/// validator runs on a thread of its own beside the execution, and level 1
/// is refused before anything is filled or proved when the program is not
/// admissible. On by default; `0` (and only `0`) turns it off, and the
/// program is then compiled and validated before the execution. The same
/// program and the same proofs either way; a verifier derives its own copy,
/// compiled and validated as before.
pub const PROGRAM_FAST_ENV: &str = "LAMBDA_VM_LEVELS_PROGRAM_FAST";

/// Whether the driver makes level 1's program the fast way:
/// [`PROGRAM_FAST_ENV`], or a test's override.
pub fn program_fast() -> bool {
    #[cfg(test)]
    if let Some(on) = *PROGRAM_FAST_FOR_TEST
        .lock()
        .unwrap_or_else(|e| e.into_inner())
    {
        return on;
    }
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        !super::wlfm_knobs::knob_word_off(std::env::var(PROGRAM_FAST_ENV).ok().as_deref())
    })
}

#[cfg(test)]
static PROGRAM_FAST_FOR_TEST: std::sync::Mutex<Option<bool>> = std::sync::Mutex::new(None);

/// `LAMBDA_VM_LEVELS_PROGRAM_POOL=<threads>`: run level 1's parallel host
/// work beside the base on a dedicated pool of this size. Unset or `0` uses
/// the existing pools. The base and the later levels keep their own scheduling.
pub const PROGRAM_POOL_ENV: &str = "LAMBDA_VM_LEVELS_PROGRAM_POOL";

fn parse_program_pool(word: Option<&str>) -> Result<usize, String> {
    match word.map(str::trim) {
        None | Some("") => Ok(0),
        Some(word) => word.parse().map_err(|_| {
            format!("{PROGRAM_POOL_ENV}={word:?}: expected a thread count (0 or greater)")
        }),
    }
}

#[cfg(test)]
std::thread_local! {
    pub(crate) static PROGRAM_POOL_FOR_TEST: std::cell::Cell<Option<usize>> = const { std::cell::Cell::new(None) };
}

fn program_pool() -> Result<Option<rayon::ThreadPool>, String> {
    #[cfg(test)]
    let threads = PROGRAM_POOL_FOR_TEST
        .get()
        .map_or_else(|| knob_word(PROGRAM_POOL_ENV, parse_program_pool), Ok)?;
    #[cfg(not(test))]
    let threads = knob_word(PROGRAM_POOL_ENV, parse_program_pool)?;
    if threads == 0 {
        return Ok(None);
    }
    rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .thread_name(|i| format!("levels-program-{i}"))
        .build()
        .map(Some)
        .map_err(|e| format!("{PROGRAM_POOL_ENV}={threads}: {e}"))
}

fn on_program_pool<R: Send>(
    pool: Option<&rayon::ThreadPool>,
    work: impl FnOnce() -> R + Send,
) -> R {
    match pool {
        Some(pool) => pool.install(work),
        None => work(),
    }
}

/// `LAMBDA_VM_LEVELS_BESIDE_NICE=1`: the host work the driver runs beside the
/// base's phase B that level 1's execution does not wait for (its stacks'
/// columns prepared on the host, [`prepare_level`], and the validator beside
/// the execution) runs below the prover's threads: the prepare on a pool of
/// its own whose threads are lowered to [`BESIDE_NICE`] (Linux), the
/// validator's thread lowered too. Phase B's host work is then scheduled
/// first and the prepare takes the cycles it leaves; on a real median the
/// prepare had ≈ 2.2 s to spare before level 1's prove needed its stacks and
/// the validator ≈ 4.3 s before the execution's end, while the two cost the
/// base's phase B ≈ 1 s on the prover's pool. Off by default (and for any word
/// but `1`). The order of the work changes, never its result.
pub const BESIDE_NICE_ENV: &str = "LAMBDA_VM_LEVELS_BESIDE_NICE";

/// The nice value [`BESIDE_NICE_ENV`]'s threads run at on Linux.
pub const BESIDE_NICE: i32 = 10;

/// `LAMBDA_VM_LEVELS_BUCKET_WAVES`: level 1's streamed execution buckets its
/// instructions by the groups each waits for, and each landed piece's wave
/// walks only the buckets that landing completes
/// ([`super::executor::StreamedExecution::new_pieced_bucketed`]). Today each
/// wave filters the whole program: on a real median, 48 M instructions, ≈ 0.1 s
/// a wave before its own work, 27 waves after the base's end; bucketed, the
/// median's recursion took 1.56 s less and the large's 5.8 s less. The same
/// instructions run in the same order each wave: the same execution and the
/// same proofs. On by default; `0` (and only `0`) turns it off.
pub const BUCKET_WAVES_ENV: &str = "LAMBDA_VM_LEVELS_BUCKET_WAVES";

/// Whether level 1's execution buckets its waves ([`BUCKET_WAVES_ENV`]).
pub fn bucket_waves() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        !super::wlfm_knobs::knob_word_off(std::env::var(BUCKET_WAVES_ENV).ok().as_deref())
    })
}

/// `LAMBDA_VM_LEVELS_SETUP_FAST`: level 1's streamed execution makes its
/// setup the fast way ([`super::executor::StreamedExecution::new_pieced_setup`]):
/// the level schedule on a thread of its own beside the need pass, and the
/// wave buckets found from each instruction's neighbour when they share
/// their groups. On a real median the setup took ≈ 2.5 s after the program,
/// and level 1's waves wait for it. The same schedule, needs and buckets:
/// the same execution and proofs. On by default; only `0` turns it off.
pub const SETUP_FAST_ENV: &str = "LAMBDA_VM_LEVELS_SETUP_FAST";

/// `LAMBDA_VM_LEVELS_FILL_BESIDE`: level 1's fill (its traces filled from
/// its execution) runs on level 1's own thread as soon as the execution ends,
/// beside the stacks' commit on the card at the base's end, where it otherwise
/// runs after that commit. Level 1's prove then starts at the later of the two
/// rather than after both. The same fill, so the same proofs. On by default;
/// only `0` turns it off. Read once a process.
pub const FILL_BESIDE_ENV: &str = "LAMBDA_VM_LEVELS_FILL_BESIDE";

std::thread_local! {
    /// A test's [`fill_beside`] on this thread.
    static FILL_BESIDE_FOR_TEST: std::cell::Cell<Option<bool>> = const { std::cell::Cell::new(None) };
}

/// Tests only: [`fill_beside`] forced on this thread (`None` restores the
/// knob). The driver reads it on its own thread.
#[doc(hidden)]
pub fn set_fill_beside_for_test(on: Option<bool>) {
    FILL_BESIDE_FOR_TEST.with(|c| c.set(on));
}

/// Whether [`FILL_BESIDE_ENV`] is on (or this thread's test override).
pub fn fill_beside() -> bool {
    if let Some(on) = FILL_BESIDE_FOR_TEST.with(std::cell::Cell::get) {
        return on;
    }
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        !super::wlfm_knobs::knob_word_off(std::env::var(FILL_BESIDE_ENV).ok().as_deref())
    })
}

/// `LAMBDA_VM_LEVELS_FILL_PARTS=1`: each level's fill runs its three parts at
/// once (the chips' traces, the lean BLAKE3 tables, the one-output selects'
/// and cap tables), each on a thread of its own, where they otherwise run one
/// after another ([`super::proof::lfm_fill_executed_parts`]). The same tables
/// from the same records, so the same proofs. `0` off; unset (or any other
/// word), on exactly where the process proves over two cards
/// ([`super::wlfm_knobs::two_cards`]), the default. The word is read once a
/// process, on the driver's thread.
pub const FILL_PARTS_ENV: &str = "LAMBDA_VM_LEVELS_FILL_PARTS";

std::thread_local! {
    /// A test's [`fill_parts`] on this thread.
    static FILL_PARTS_FOR_TEST: std::cell::Cell<Option<bool>> = const { std::cell::Cell::new(None) };
}

/// Tests only: [`fill_parts`] forced on this thread (`None` restores the
/// knob). The driver reads it on its own thread.
#[doc(hidden)]
pub fn set_fill_parts_for_test(on: Option<bool>) {
    FILL_PARTS_FOR_TEST.with(|c| c.set(on));
}

/// Whether [`FILL_PARTS_ENV`] is on (or this thread's test override).
pub fn fill_parts() -> bool {
    if let Some(on) = FILL_PARTS_FOR_TEST.with(std::cell::Cell::get) {
        return on;
    }
    static WORD: std::sync::OnceLock<Option<bool>> = std::sync::OnceLock::new();
    WORD.get_or_init(|| {
        super::wlfm_knobs::knob_word_two_card(std::env::var(FILL_PARTS_ENV).ok().as_deref())
    })
    .unwrap_or_else(super::wlfm_knobs::two_cards)
}

/// `LAMBDA_VM_LEVELS_COMMIT_NEXT_EARLY=1`: when level 2 executes beside level
/// 1's prove ([`stream_next`]), its prepared stacks are committed on the card
/// the moment level 1's prove returns, beside the end of level 2's execution
/// and its fill on the host, where they are otherwise committed after that
/// fill. The same commit of the same columns, earlier: the same proofs. `0`
/// off; unset (or any other word), on exactly where the process proves over
/// two cards ([`super::wlfm_knobs::two_cards`]), the default. The word is read
/// once a process, on the driver's thread.
pub const COMMIT_NEXT_EARLY_ENV: &str = "LAMBDA_VM_LEVELS_COMMIT_NEXT_EARLY";

std::thread_local! {
    /// A test's [`commit_next_early`] on this thread.
    static COMMIT_NEXT_EARLY_FOR_TEST: std::cell::Cell<Option<bool>> = const { std::cell::Cell::new(None) };
}

/// Tests only: [`commit_next_early`] forced on this thread (`None` restores
/// the knob). The driver reads it on its own thread.
#[doc(hidden)]
pub fn set_commit_next_early_for_test(on: Option<bool>) {
    COMMIT_NEXT_EARLY_FOR_TEST.with(|c| c.set(on));
}

/// Whether [`COMMIT_NEXT_EARLY_ENV`] is on (or this thread's test override).
pub fn commit_next_early() -> bool {
    if let Some(on) = COMMIT_NEXT_EARLY_FOR_TEST.with(std::cell::Cell::get) {
        return on;
    }
    static WORD: std::sync::OnceLock<Option<bool>> = std::sync::OnceLock::new();
    WORD.get_or_init(|| {
        super::wlfm_knobs::knob_word_two_card(std::env::var(COMMIT_NEXT_EARLY_ENV).ok().as_deref())
    })
    .unwrap_or_else(super::wlfm_knobs::two_cards)
}

/// `LAMBDA_VM_LEVELS_PLAN_LEAN`: level 1's plan, made by the prover from its
/// own prepared roots, skips measuring each group's chip rows by emitting it
/// alone ([`WhirBlockPlan::chip_loads`]): a level's leaf count is fixed at
/// one, so nothing reads them. The program is emitted once, over every
/// group, either way: the same proofs. A verifier's plan always measures.
/// On by default, `0` turns it off. Read once a process, on the driver's thread.
pub const PLAN_LEAN_ENV: &str = "LAMBDA_VM_LEVELS_PLAN_LEAN";

std::thread_local! {
    /// A test's [`plan_lean`] on this thread.
    static PLAN_LEAN_FOR_TEST: std::cell::Cell<Option<bool>> = const { std::cell::Cell::new(None) };
}

/// Tests only: [`plan_lean`] forced on this thread (`None` restores the
/// knob). The driver reads it on its own thread.
#[doc(hidden)]
pub fn set_plan_lean_for_test(on: Option<bool>) {
    PLAN_LEAN_FOR_TEST.with(|c| c.set(on));
}

/// Whether [`PLAN_LEAN_ENV`] is on (or this thread's test override).
pub fn plan_lean() -> bool {
    if let Some(on) = PLAN_LEAN_FOR_TEST.with(std::cell::Cell::get) {
        return on;
    }
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        !super::wlfm_knobs::knob_word_off(std::env::var(PLAN_LEAN_ENV).ok().as_deref())
    })
}

/// `LAMBDA_VM_LEVELS_STATEMENT_EARLY`: the base predicts its statement
/// once every table's shape is known, before its last commits
/// (`block_whir::prove_block_whir_local_observed_early`), and level 1 makes
/// its plan and program from the prediction while phase A ends. They are used
/// only if the final statement equals the prediction in every field and the
/// prepared roots are equal; else they are made again from the final
/// statement, as without the knob. Nothing of level 1 is published before.
/// The same program from an equal statement: the same proofs. Acts only
/// where level 1 runs beside the base with its program validated beside its
/// execution. On by default, `0` turns it off. Read once a process, on the
/// driver's thread.
pub const STATEMENT_EARLY_ENV: &str = "LAMBDA_VM_LEVELS_STATEMENT_EARLY";

std::thread_local! {
    /// A test's [`statement_early`] on this thread.
    static STATEMENT_EARLY_FOR_TEST: std::cell::Cell<Option<bool>> = const { std::cell::Cell::new(None) };
}

/// Tests only: [`statement_early`] forced on this thread (`None` restores the
/// knob). The driver reads it on its own thread.
#[doc(hidden)]
pub fn set_statement_early_for_test(on: Option<bool>) {
    STATEMENT_EARLY_FOR_TEST.with(|c| c.set(on));
}

/// Whether [`STATEMENT_EARLY_ENV`] is on (or this thread's test override).
pub fn statement_early() -> bool {
    if let Some(on) = STATEMENT_EARLY_FOR_TEST.with(std::cell::Cell::get) {
        return on;
    }
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        !super::wlfm_knobs::knob_word_off(std::env::var(STATEMENT_EARLY_ENV).ok().as_deref())
    })
}

/// A statement and its prepared roots, as the base hands them to level 1.
type StatementWithRoots = (
    crate::block_whir::OwnedBlockStatement,
    Vec<crate::block_whir::PreparedRoots>,
);

/// What reaches level 1's thread from the base: the predicted statement
/// (with the driver's clock when it came), then the final one; or the final
/// one alone.
enum Stated {
    Early(StatementWithRoots, f64),
    Final(StatementWithRoots),
}

/// Level 1's plan and program, and their stamps, made from one statement.
struct Made {
    plan: WhirBlockPlan,
    planned: f64,
    plan_split: Option<[f64; 6]>,
    program: LfmProgram,
    marks: Vec<u32>,
    walk: f64,
    compiled: f64,
    compile_split: Option<[f64; 3]>,
}

/// What level 1 made of the base's early statement, on the driver's clock:
/// when the statement came, when its plan and program were made, when the
/// final statement came and how long level 1 waited for it, and whether the
/// early program was used.
#[derive(Clone, Copy, Debug, Default)]
pub struct EarlyNote {
    pub stated_at: f64,
    pub made_at: f64,
    pub confirmed_at: f64,
    pub waited: f64,
    pub used: bool,
}

/// Early statements level 1 could not use, since the process started.
static EARLY_MISMATCHES: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

/// [`EARLY_MISMATCHES`]' count.
pub fn early_mismatches() -> usize {
    EARLY_MISMATCHES.load(std::sync::atomic::Ordering::Relaxed)
}

/// `LAMBDA_VM_LEVELS_WAVES_IN_ORDER=1`: level 1's and level 2's streamed
/// executions run each wave's instructions in program order on the wave's own
/// thread, hashing inline ([`super::executor::StreamedExecution::new_pieced_in_order`]),
/// where each wave otherwise walks the whole program's level schedule with its
/// hashes on the shared pool. The same instructions a wave, in an order that
/// respects every dependency, so the same executions and proofs. Acts only on
/// bucketed waves ([`bucket_waves`], on by default). `0` off; unset (or any
/// other word), on exactly where the process proves over two cards
/// ([`super::wlfm_knobs::two_cards`]), the default. The word is read once a
/// process, on the driver's thread.
pub const WAVES_IN_ORDER_ENV: &str = "LAMBDA_VM_LEVELS_WAVES_IN_ORDER";

std::thread_local! {
    /// A test's [`waves_in_order`] on this thread.
    static WAVES_IN_ORDER_FOR_TEST: std::cell::Cell<Option<bool>> = const { std::cell::Cell::new(None) };
}

/// Tests only: [`waves_in_order`] forced on this thread (`None` restores the
/// knob). The driver reads it on its own thread.
#[doc(hidden)]
pub fn set_waves_in_order_for_test(on: Option<bool>) {
    WAVES_IN_ORDER_FOR_TEST.with(|c| c.set(on));
}

/// Whether [`WAVES_IN_ORDER_ENV`] is on (or this thread's test override).
pub fn waves_in_order() -> bool {
    if let Some(on) = WAVES_IN_ORDER_FOR_TEST.with(std::cell::Cell::get) {
        return on;
    }
    static WORD: std::sync::OnceLock<Option<bool>> = std::sync::OnceLock::new();
    WORD.get_or_init(|| {
        super::wlfm_knobs::knob_word_two_card(std::env::var(WAVES_IN_ORDER_ENV).ok().as_deref())
    })
    .unwrap_or_else(super::wlfm_knobs::two_cards)
}

/// How level 2's streamed execution runs, as the driver read its knobs: its
/// fill's parts at once ([`fill_parts`]), its waves in program order
/// ([`waves_in_order`]).
#[derive(Clone, Copy, Debug, Default)]
pub(super) struct StreamOpts {
    pub(super) fill_parts: bool,
    pub(super) in_order: bool,
}

/// Level 1's execution as its thread hands it on: executed, for the driver to
/// fill after the stacks' commit, or already filled beside it
/// ([`fill_beside`]).
#[allow(clippy::large_enum_variant)] // moved once, from level 1's thread to the driver
enum Level1Out {
    Executed(
        super::executor::LfmExecution,
        Option<(
            Result<super::b3l_tables::LeanPrefilled, String>,
            super::b3l_tables::LeanBesideStats,
            f64,
        )>,
    ),
    Filled {
        filled: super::proof::LfmFilled,
        split: super::executor::ExecSplit,
        lean: Option<(super::b3l_tables::LeanBesideStats, f64)>,
        lean_prefilled: bool,
        fill_split: (f64, f64),
        lean_parts: (f64, f64),
        /// The fill's seconds and when it ended (the driver's clock).
        fill: f64,
        done: f64,
    },
}

/// Level 1's execution as the driver takes it after the base: to fill, or
/// filled ([`Level1Out`]).
#[allow(clippy::large_enum_variant)] // moved once, from level 1's thread to the driver
enum Level1Ready {
    Executed(
        super::executor::LfmExecution,
        Option<super::b3l_tables::LeanPrefilled>,
    ),
    Filled(super::proof::LfmFilled, f64),
}

/// Whether level 1's execution makes its setup the fast way ([`SETUP_FAST_ENV`]).
pub fn setup_fast() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        !super::wlfm_knobs::knob_word_off(std::env::var(SETUP_FAST_ENV).ok().as_deref())
    })
}

/// How level 1's streamed execution met the base's groups, on the driver's
/// clock: what it ran before any group (its setup and first wave), when the
/// front landed and its wave, when each group's share arrived from phase B,
/// and when each piece landed with its wave's seconds. A report: nothing reads
/// it but the harness.
#[derive(Clone, Debug, Default)]
pub struct Level1Landing {
    /// Seconds from the execution's start to its setup's end (the first wave,
    /// the instructions that read no arena, included).
    pub setup: f64,
    /// When the front landed, and its wave's seconds.
    pub front: (f64, f64),
    /// When each group's share arrived, in arrival order: `(group, at)`.
    pub arrivals: Vec<(usize, f64)>,
    /// When each piece landed, in landing order: `(piece, at, wave seconds)`.
    pub landed: Vec<(usize, f64, f64)>,
    /// The groups whose arena words the landing thread made, and the seconds
    /// it took ([`super::whir_block::group_arena_words`]).
    pub words: (usize, f64),
    /// Where the execution ran, under `LAMBDA_VM_LEVELS_PLACEMENT`.
    pub placement: Option<super::placement::Placement>,
}

/// Whether the driver lowers its beside work ([`BESIDE_NICE_ENV`]), or a
/// test's override.
pub fn beside_nice() -> bool {
    #[cfg(test)]
    if let Some(on) = *BESIDE_NICE_FOR_TEST
        .lock()
        .unwrap_or_else(|e| e.into_inner())
    {
        return on;
    }
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        super::wlfm_knobs::knob_word_on(std::env::var(BESIDE_NICE_ENV).ok().as_deref())
    })
}

#[cfg(test)]
static BESIDE_NICE_FOR_TEST: std::sync::Mutex<Option<bool>> = std::sync::Mutex::new(None);

/// A test's [`beside_nice`] for every thread until the guard drops.
#[cfg(test)]
pub(crate) fn set_beside_nice_for_test(on: bool) -> BesideNiceForTest {
    let before = BESIDE_NICE_FOR_TEST
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .replace(on);
    BesideNiceForTest(before)
}

#[cfg(test)]
pub(crate) struct BesideNiceForTest(Option<bool>);

#[cfg(test)]
impl Drop for BesideNiceForTest {
    fn drop(&mut self) {
        *BESIDE_NICE_FOR_TEST
            .lock()
            .unwrap_or_else(|e| e.into_inner()) = self.0;
    }
}

/// The beside work's own pool ([`BESIDE_NICE_ENV`]): as many threads as the
/// prover's pool, each lowered to [`BESIDE_NICE`]. Built once a process;
/// `None` if it cannot be (the work then runs on the prover's pool).
fn beside_pool() -> Option<&'static rayon::ThreadPool> {
    static POOL: std::sync::OnceLock<Option<rayon::ThreadPool>> = std::sync::OnceLock::new();
    POOL.get_or_init(|| {
        rayon::ThreadPoolBuilder::new()
            .num_threads(rayon::current_num_threads())
            .thread_name(|i| format!("levels-beside-{i}"))
            .start_handler(|_| super::whir_block_tree::lower_this_threads_priority(BESIDE_NICE))
            .build()
            .ok()
    })
    .as_ref()
}

/// [`prepare_level`] as the driver runs it beside the base: on the beside pool
/// under [`beside_nice`], on the prover's pool otherwise. The same prebuild.
/// The driver calls [`prepare_level_watching`]; tests call this.
#[cfg(test)]
pub(crate) fn prepare_level_beside(
    program: &LfmProgram,
    family: WlfmFamily,
    rounds: Option<u8>,
) -> Result<WlfmPrebuild, String> {
    prepare_level_beside_as(program, level_levers(family), rounds)
}

/// A level's host part as the driver runs it beside the base, under
/// `levers`: on the beside pool under [`beside_nice`], on the prover's pool
/// otherwise. The same prebuild.
#[cfg(test)]
pub(crate) fn prepare_level_beside_as(
    program: &LfmProgram,
    levers: WlfmLevers,
    rounds: Option<u8>,
) -> Result<WlfmPrebuild, String> {
    prepare_level_watching(program, levers, rounds, None, None).map(|(pre, _)| pre)
}

/// [`prepare_level_beside_as`], watching `base_returned` when given
/// ([`super::wlfm::BUILDER_WHOLE_POOL_ENV`]): raised, the columns not yet
/// made and the packs go to the whole pool. The same prebuild, and what the
/// watch did.
fn prepare_level_watching(
    program: &LfmProgram,
    levers: WlfmLevers,
    rounds: Option<u8>,
    base_returned: Option<std::sync::Arc<std::sync::atomic::AtomicBool>>,
    program_pool: Option<&rayon::ThreadPool>,
) -> Result<(WlfmPrebuild, Option<super::wlfm::WholePoolNote>), String> {
    // On the thread that runs the prepare (a pool's worker under the beside
    // pool): the watch and its note are that thread's.
    let run = move || {
        let _watch = base_returned.map(super::wlfm::watch_base);
        let _ = super::wlfm::take_whole_pool_note();
        let pre = prepare_level_packs_as(
            program,
            levers,
            rounds,
            packs_beside(),
            program_pool.is_some(),
        )?;
        Ok((pre, super::wlfm::take_whole_pool_note()))
    };
    let pool = program_pool.or_else(|| beside_nice().then(beside_pool).flatten());
    on_program_pool(pool, run)
}

/// A test's [`program_fast`] for every thread until the guard drops.
#[cfg(test)]
pub(crate) fn set_program_fast_for_test(on: bool) -> ProgramFastForTest {
    let before = PROGRAM_FAST_FOR_TEST
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .replace(on);
    ProgramFastForTest(before)
}

#[cfg(test)]
pub(crate) struct ProgramFastForTest(Option<bool>);

#[cfg(test)]
impl Drop for ProgramFastForTest {
    fn drop(&mut self) {
        *PROGRAM_FAST_FOR_TEST
            .lock()
            .unwrap_or_else(|e| e.into_inner()) = self.0;
    }
}

/// `LAMBDA_VM_LEVELS_VALIDATE_AT_ONCE=1`: level 1's admission validator runs
/// its five checks at once, each on a thread of its own
/// ([`super::validator::validate_at_once`]), where they otherwise run one
/// after another on one thread. The same checks, the same verdict (of several
/// failures, the earliest check's). Off by default (and for any word but `1`).
/// Read once a process.
pub const VALIDATE_AT_ONCE_ENV: &str = "LAMBDA_VM_LEVELS_VALIDATE_AT_ONCE";

/// A test's [`validate_at_once`], process wide (0 the knob, 1 off, 2 on).
static VALIDATE_AT_ONCE_FOR_TEST: std::sync::atomic::AtomicU8 = std::sync::atomic::AtomicU8::new(0);

/// Tests only: [`validate_at_once`] forced for the process (`None` restores
/// the knob).
#[doc(hidden)]
pub fn set_validate_at_once_for_test(on: Option<bool>) {
    let v = match on {
        None => 0,
        Some(false) => 1,
        Some(true) => 2,
    };
    VALIDATE_AT_ONCE_FOR_TEST.store(v, std::sync::atomic::Ordering::Relaxed);
}

/// Whether [`VALIDATE_AT_ONCE_ENV`] is on (or a test's override).
pub fn validate_at_once() -> bool {
    match VALIDATE_AT_ONCE_FOR_TEST.load(std::sync::atomic::Ordering::Relaxed) {
        1 => return false,
        2 => return true,
        _ => {}
    }
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        super::wlfm_knobs::knob_word_on(std::env::var(VALIDATE_AT_ONCE_ENV).ok().as_deref())
    })
}

/// `LAMBDA_VM_LEVELS_VALIDATE_AFTER_FILL=1`: level 1's admission validator,
/// run beside the execution ([`program_fast`]), is joined after level 1's
/// fill beside the base ([`fill_beside`]) instead of before it, so the fill
/// no longer waits for the verdict. The verdict is still read on level 1's
/// thread before it hands anything to the prove: an inadmissible program is
/// refused before a byte of level 1 is proved. Without [`program_fast`] or
/// [`fill_beside`] it does nothing. Off by default (and for any word but
/// `1`). Read once a process.
pub const VALIDATE_AFTER_FILL_ENV: &str = "LAMBDA_VM_LEVELS_VALIDATE_AFTER_FILL";

/// A test's [`validate_after_fill`], process wide (0 the knob, 1 off, 2 on).
static VALIDATE_AFTER_FILL_FOR_TEST: std::sync::atomic::AtomicU8 =
    std::sync::atomic::AtomicU8::new(0);

/// Tests only: [`validate_after_fill`] forced for the process (`None`
/// restores the knob).
#[doc(hidden)]
pub fn set_validate_after_fill_for_test(on: Option<bool>) {
    let v = match on {
        None => 0,
        Some(false) => 1,
        Some(true) => 2,
    };
    VALIDATE_AFTER_FILL_FOR_TEST.store(v, std::sync::atomic::Ordering::Relaxed);
}

/// Whether [`VALIDATE_AFTER_FILL_ENV`] is on (or a test's override).
pub fn validate_after_fill() -> bool {
    match VALIDATE_AFTER_FILL_FOR_TEST.load(std::sync::atomic::Ordering::Relaxed) {
        1 => return false,
        2 => return true,
        _ => {}
    }
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        super::wlfm_knobs::knob_word_on(std::env::var(VALIDATE_AFTER_FILL_ENV).ok().as_deref())
    })
}

/// A test's fault in level 1's emitted program (process wide): a constant's
/// count raised by one, which the admission validator refuses
/// ([`set_level1_program_fault_for_test`]).
#[cfg(test)]
static LEVEL1_PROGRAM_FAULT_FOR_TEST: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(false);

/// Tests only: level 1's emitted program gets a constant's count raised by
/// one (`false` restores today's), so its admission must refuse it.
#[cfg(test)]
pub(crate) fn set_level1_program_fault_for_test(on: bool) {
    LEVEL1_PROGRAM_FAULT_FOR_TEST.store(on, std::sync::atomic::Ordering::Relaxed);
}

/// `program`, faulted when a test asked ([`set_level1_program_fault_for_test`]).
#[cfg(test)]
fn level1_program_fault_for_test(mut program: LfmProgram) -> LfmProgram {
    if LEVEL1_PROGRAM_FAULT_FOR_TEST.load(std::sync::atomic::Ordering::Relaxed)
        && let Some(super::instr::Instr::Const { mult, .. }) = program
            .instrs
            .iter_mut()
            .find(|i| matches!(i, super::instr::Instr::Const { .. }))
    {
        *mult += 1;
    }
    program
}

/// Level 1's admission validator over `program`: its five checks in series,
/// or at once under [`VALIDATE_AT_ONCE_ENV`].
fn validate_level(program: &LfmProgram) -> Result<(), super::validator::LfmViolation> {
    if validate_at_once() {
        super::validator::validate_at_once(program)
    } else {
        super::validator::validate(program)
    }
}

/// The admission validator over `program` on a thread of `scope`'s: its
/// seconds, or why the program is not admissible.
pub(crate) fn spawn_validation<'scope, 'env>(
    scope: &'scope std::thread::Scope<'scope, 'env>,
    program: &'env LfmProgram,
) -> std::thread::ScopedJoinHandle<'scope, Result<f64, String>> {
    scope.spawn(move || {
        if beside_nice() {
            super::whir_block_tree::lower_this_threads_priority(BESIDE_NICE);
        }
        let t = std::time::Instant::now();
        validate_level(program)
            .map_err(|e| format!("level 1's program is not admissible: {e:?}"))?;
        Ok(t.elapsed().as_secs_f64())
    })
}

/// [`spawn_validation`]'s verdict: its seconds, or the refusal.
pub(crate) fn joined_validation(
    handle: std::thread::ScopedJoinHandle<'_, Result<f64, String>>,
) -> Result<f64, String> {
    handle
        .join()
        .map_err(|_| "level 1's validator panicked".to_string())?
}

/// A level's host part ([`prepare_wlfm`]), its chip prefixes packed with it
/// when it holds its chips as records: nothing here waits for the card.
#[cfg(test)]
pub(crate) fn prepare_level(
    program: &LfmProgram,
    family: WlfmFamily,
    rounds: Option<u8>,
) -> Result<WlfmPrebuild, String> {
    prepare_level_as(program, level_levers(family), rounds)
}

/// A level's host part ([`prepare_wlfm`]) under `levers` (every level's, or
/// the last level's own, [`levers_for`]), its chip prefixes packed with it
/// when it holds its chips as records: nothing here waits for the card.
pub(crate) fn prepare_level_as(
    program: &LfmProgram,
    levers: WlfmLevers,
    rounds: Option<u8>,
) -> Result<WlfmPrebuild, String> {
    prepare_level_packs_as(program, levers, rounds, packs_beside(), false)
}

/// [`prepare_level`] with its two packs made at once under `beside`
/// ([`PACKS_BESIDE_ENV`]), every level's levers.
#[cfg(test)]
pub(crate) fn prepare_level_packs(
    program: &LfmProgram,
    family: WlfmFamily,
    rounds: Option<u8>,
    beside: bool,
) -> Result<WlfmPrebuild, String> {
    prepare_level_packs_as(program, level_levers(family), rounds, beside, false)
}

/// [`prepare_level_as`] with its two packs made at once under `beside`
/// ([`PACKS_BESIDE_ENV`]): the chip prefixes on this thread, the narrow packs
/// on their pool from a thread of their own, both read the stacks' columns
/// and write nothing the other reads; then both installed. The same
/// prebuild. Only when both are made; otherwise as today.
fn prepare_level_packs_as(
    program: &LfmProgram,
    levers: WlfmLevers,
    rounds: Option<u8>,
    beside: bool,
    on_dedicated_pool: bool,
) -> Result<WlfmPrebuild, String> {
    let hasher = program.hasher(<BlockFamily as ProofFamily>::SOCKET);
    let mut pre = prepare_wlfm_rounds(program, hasher, LEVEL_RATE, levers, rounds)?;
    // Under `LAMBDA_VM_LEVELS_BUILDER_WHOLE_POOL`, once the base has
    // returned, the narrow packs leave their two threads for the caller's
    // pool (the same packs).
    let packs_whole = super::wlfm::base_returned_seen();
    let pack_pool = || (!packs_whole).then(narrow_pack_pool).flatten();
    if super::wlfm::narrow_commit() && super::wlfm::base_watched() {
        super::wlfm::note_whole_pool_packs(packs_whole);
    }
    if beside && super::wlfm::chip_records() && super::wlfm::narrow_commit() {
        let (prefixes, packs) = if on_dedicated_pool && pack_pool().is_none() {
            // A fresh std thread would send these packs to the global pool.
            rayon::join(|| pre.chip_prefixes_made(), || pre.narrow_packs_made())
        } else {
            std::thread::scope(|scope| {
                let pre = &pre;
                let packs =
                    scope.spawn(move || on_program_pool(pack_pool(), || pre.narrow_packs_made()));
                let prefixes = pre.chip_prefixes_made();
                let packs = packs
                    .join()
                    .unwrap_or_else(|panic| std::panic::resume_unwind(panic));
                (prefixes, packs)
            })
        };
        pre.install_chip_prefixes(prefixes?);
        pre.install_narrow_packs(packs);
        return Ok(pre);
    }
    if super::wlfm::chip_records() {
        pre.hold_chip_prefixes()?;
    }
    if super::wlfm::narrow_commit() {
        match pack_pool() {
            Some(pool) => pool.install(|| pre.hold_narrow_packs()),
            None => pre.hold_narrow_packs(),
        }
    }
    Ok(pre)
}

/// `LAMBDA_VM_LEVELS_PACKS_BESIDE=1`: a level's host part makes its chip
/// prefixes and its stacks' narrow packs at once ([`prepare_level_packs`]),
/// where it otherwise makes the prefixes, then the packs. The same packs. Off
/// by default (and for any word but `1`); read once a process.
pub const PACKS_BESIDE_ENV: &str = "LAMBDA_VM_LEVELS_PACKS_BESIDE";

/// A test's [`packs_beside`], process wide (0 the knob, 1 off, 2 on).
static PACKS_BESIDE_FOR_TEST: std::sync::atomic::AtomicU8 = std::sync::atomic::AtomicU8::new(0);

/// Tests only: [`packs_beside`] forced for the whole process (`None` restores
/// the knob): a level's prepare runs on threads of the driver's own. A test
/// that sets it runs alone.
#[doc(hidden)]
pub fn set_packs_beside_for_test(on: Option<bool>) {
    let v = match on {
        None => 0,
        Some(false) => 1,
        Some(true) => 2,
    };
    PACKS_BESIDE_FOR_TEST.store(v, std::sync::atomic::Ordering::Relaxed);
}

/// Whether [`PACKS_BESIDE_ENV`] is on (or a test's override).
pub fn packs_beside() -> bool {
    match PACKS_BESIDE_FOR_TEST.load(std::sync::atomic::Ordering::Relaxed) {
        1 => return false,
        2 => return true,
        _ => {}
    }
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        super::wlfm_knobs::knob_word_on(std::env::var(PACKS_BESIDE_ENV).ok().as_deref())
    })
}

/// The threads a level's host part packs its prepared stacks narrow on
/// ([`super::wlfm::WlfmPrebuild::hold_narrow_packs`]), beside the base's
/// phase B: two, not the shared pool, which phase B binds (a pooled gather
/// there cost phase B about one for one).
const NARROW_PACK_THREADS: usize = 2;

/// [`NARROW_PACK_THREADS`]' pool, made once; `None` if it cannot be (the
/// pack then runs on the caller's pool).
fn narrow_pack_pool() -> Option<&'static rayon::ThreadPool> {
    static POOL: std::sync::OnceLock<Option<rayon::ThreadPool>> = std::sync::OnceLock::new();
    POOL.get_or_init(|| {
        rayon::ThreadPoolBuilder::new()
            .num_threads(NARROW_PACK_THREADS)
            .thread_name(|k| format!("narrow-pack-{k}"))
            .build()
            .ok()
    })
    .as_ref()
}

/// A prepared level committed where [`stack_place`] puts it; under
/// [`STACKS_BY_CARD_ENV`] (with the levels' split ready and the stacks on a
/// card) each group's stack on the card its group is assigned to
/// ([`stark::phase_b_split::assign_cards`] of [`WlfmPrebuild::group_prices`]).
pub(crate) fn commit_level(pre: WlfmPrebuild, level: usize) -> Result<WlfmBuild, String> {
    let place = stack_place(pre.prepared_cells(), level, stacks_forced()?);
    let cards = (place != StackPlace::Host && stacks_by_card_acts())
        .then(|| stark::phase_b_split::assign_cards(&pre.group_prices(), &[]))
        .filter(|cards| cards.contains(&1));
    if let Some(cards) = &cards {
        let cells = pre.prepared_cells_by_group();
        let mut by_card = [(0u64, 0usize); 2];
        for (g, &c) in cards.iter().enumerate() {
            let at = &mut by_card[usize::from(c.min(1))];
            let cells = cells.get(g).copied().unwrap_or(0);
            at.0 += cells.saturating_mul(STACK_CARD_BYTES_PER_CELL);
            at.1 += usize::from(cells > 0);
        }
        let gib = |b: u64| b as f64 / (1u64 << 30) as f64;
        eprintln!(
            "LEVELS STACKS level {level} by card (LAMBDA_VM_LEVELS_STACKS_BY_CARD): card 0 {:.2} GiB in \
             {} stacks · card 1 {:.2} GiB in {} stacks · {}",
            gib(by_card[0].0),
            by_card[0].1,
            gib(by_card[1].0),
            by_card[1].1,
            match place {
                StackPlace::Card => "held",
                _ => "retired",
            }
        );
    }
    pre.commit_on(place, cards)
}

/// `LAMBDA_VM_LEVELS_STACKS_BY_CARD`: under the levels' split
/// (`LAMBDA_VM_TWO_CARD_LEVELS`, its second card ready), a level's groups are
/// assigned to the two cards before its prepared stacks are committed, each
/// stack is committed on its group's card (both cards at once), and phase B
/// argues and opens each group whose stack is held on the card that holds it.
/// Where a held stack otherwise pins its group to card 0, a level whose stacks
/// are all held then splits too. The same roots on either card: the same
/// proofs. On by default, `0` turns it off. Read once a process;
/// a test's override is the process's (level 2's commit runs on a thread of
/// its own).
pub const STACKS_BY_CARD_ENV: &str = "LAMBDA_VM_LEVELS_STACKS_BY_CARD";

/// A test's [`stacks_by_card`], process wide (0 the knob, 1 off, 2 on).
static STACKS_BY_CARD_FOR_TEST: std::sync::atomic::AtomicU8 = std::sync::atomic::AtomicU8::new(0);

/// Tests only: [`stacks_by_card`] forced for the process (`None` restores the
/// knob).
#[doc(hidden)]
pub fn set_stacks_by_card_for_test(on: Option<bool>) {
    let v = match on {
        None => 0,
        Some(false) => 1,
        Some(true) => 2,
    };
    STACKS_BY_CARD_FOR_TEST.store(v, std::sync::atomic::Ordering::Relaxed);
}

/// Whether [`STACKS_BY_CARD_ENV`] is on (or a test's override).
pub fn stacks_by_card() -> bool {
    match STACKS_BY_CARD_FOR_TEST.load(std::sync::atomic::Ordering::Relaxed) {
        1 => return false,
        2 => return true,
        _ => {}
    }
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        !super::wlfm_knobs::knob_word_off(std::env::var(STACKS_BY_CARD_ENV).ok().as_deref())
    })
}

/// Whether [`stacks_by_card`] acts: the knob on, the levels' split on and its
/// second card ready.
fn stacks_by_card_acts() -> bool {
    stacks_by_card()
        && multilinear::gpu::two_card_levels_on()
        && multilinear::gpu::two_card_split_ready()
}

/// Whether a chain whose last level (its `count`-th) built `last` goes on.
fn level_more(stop: LevelStop, count: usize, last: &WlfmBuild) -> bool {
    match stop {
        LevelStop::Levels(n) => count < n.clamp(1, MAX_LEVELS),
        LevelStop::Groups(g) => count < MAX_LEVELS && last.artifacts.sizes.len() > g,
    }
}

/// The level after `child`'s, prepared on the host: its program (the W-leg
/// over `child`), where its arena's pieces end (the front, then each of the
/// child's groups), and its stacks' columns, uncommitted, its chains under the
/// child's round form (every level's is the base's, [`level_rounds`]).
fn prepare_next(
    child: WlfmChild,
    family: WlfmFamily,
    stop: LevelStop,
    index: usize,
    last: LastLevel,
) -> Result<PreparedNext, String> {
    let (program, marks) = next_program(&child)?;
    let rounds = child.artifacts.rounds;
    let pre = prepare_as_level(stop, index, last, family, |levers| {
        prepare_level_as(&program, levers, rounds)
    })?;
    Ok((program, pre, marks))
}

/// The level after `child`'s: its program and where its arena's pieces end.
fn next_program(child: &WlfmChild) -> Result<(LfmProgram, Vec<u32>), String> {
    wlfm_leg_program_marked(child)
}

/// `LAMBDA_VM_LEVELS_NEXT_BESIDE=1`: when level 2 executes beside level 1's
/// prove ([`stream_next`]), its execution is set up and streamed as soon as
/// its program exists, its host preparation (columns, stacks, packs) on a
/// thread of its own beside it ([`stream_next_level_beside`]), where today the
/// execution waits for that preparation. The same program, prebuild and
/// execution: the same proofs. `0` off; unset (or any other word), on exactly
/// where the process proves over two cards
/// ([`super::wlfm_knobs::two_cards`]), the default. The word is read once a
/// process, on the driver's thread.
pub const NEXT_BESIDE_ENV: &str = "LAMBDA_VM_LEVELS_NEXT_BESIDE";

std::thread_local! {
    /// A test's [`next_beside`] on this thread.
    static NEXT_BESIDE_FOR_TEST: std::cell::Cell<Option<bool>> = const { std::cell::Cell::new(None) };
}

/// Tests only: [`next_beside`] forced on this thread (`None` restores the
/// knob). The driver reads it on its own thread.
#[doc(hidden)]
pub fn set_next_beside_for_test(on: Option<bool>) {
    NEXT_BESIDE_FOR_TEST.with(|c| c.set(on));
}

/// Whether [`NEXT_BESIDE_ENV`] is on (or this thread's test override).
pub fn next_beside() -> bool {
    if let Some(on) = NEXT_BESIDE_FOR_TEST.with(std::cell::Cell::get) {
        return on;
    }
    static WORD: std::sync::OnceLock<Option<bool>> = std::sync::OnceLock::new();
    WORD.get_or_init(|| {
        super::wlfm_knobs::knob_word_two_card(std::env::var(NEXT_BESIDE_ENV).ok().as_deref())
    })
    .unwrap_or_else(super::wlfm_knobs::two_cards)
}

/// [`prepare_next`]'s level: its program, its prebuild and its arena's marks.
type PreparedNext = (LfmProgram, WlfmPrebuild, Vec<u32>);

/// `LAMBDA_VM_LEVELS_STREAM_NEXT`: level 2's execution streams over level 1's
/// prove, beside it: the arena's front lands as soon as level 1 has opened its
/// first group, each group's words as level 1 opens that group
/// ([`super::wlfm::prove_wlfm_traces_observed`],
/// [`super::wlfm_leg::wlfm_leg_group`]), each running the instructions that
/// read it; level 2 fills at its last piece. Without it, level 2 executes over
/// the arena written from level 1's whole proof once that proof is done. The
/// same words, so the same execution and proofs. On by default; only `0` turns
/// it off. Read once a process.
pub const STREAM_NEXT_ENV: &str = "LAMBDA_VM_LEVELS_STREAM_NEXT";

std::thread_local! {
    /// A test's [`stream_next`] on this thread.
    static STREAM_NEXT_FOR_TEST: std::cell::Cell<Option<bool>> = const { std::cell::Cell::new(None) };
}

/// Tests only: [`stream_next`] forced on this thread (`None` restores the
/// knob). The driver reads it on its own thread.
#[doc(hidden)]
pub fn set_stream_next_for_test(on: Option<bool>) {
    STREAM_NEXT_FOR_TEST.with(|c| c.set(on));
}

/// Whether [`STREAM_NEXT_ENV`] is on (or this thread's test override).
pub fn stream_next() -> bool {
    if let Some(on) = STREAM_NEXT_FOR_TEST.with(std::cell::Cell::get) {
        return on;
    }
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        !super::wlfm_knobs::knob_word_off(std::env::var(STREAM_NEXT_ENV).ok().as_deref())
    })
}

/// Level 2's program and prebuild, and its execution filled beside level 1's
/// prove when it streamed ([`stream_next`]). No prebuild when it was handed
/// on to be committed early ([`commit_next_early`]).
type StreamedNext = (
    LfmProgram,
    Option<WlfmPrebuild>,
    Option<(super::proof::LfmFilled, NextStream)>,
);

/// When level 2's streamed execution ran (the driver's clock): its start, its
/// first and last pieces' landing, its end, and its fill's seconds; and its
/// pieces.
#[derive(Clone, Copy, Debug, Default)]
pub struct NextStream {
    pub pieces: usize,
    pub from: f64,
    pub first: f64,
    pub last: f64,
    pub executed: f64,
    pub fill: f64,
    /// Under [`next_beside`], when its prebuild was made beside the execution
    /// (`None` on today's path, where it was made before the execution).
    pub prebuilt: Option<f64>,
}

/// When level 2's stacks' commit started (the driver's clock), whether early
/// ([`commit_next_early`]: as level 1's prove returned), and when the driver
/// joined level 2's streamed execution (`None` when it did not stream).
#[derive(Clone, Copy, Debug, Default)]
pub struct NextCommit {
    pub from: f64,
    pub early: bool,
    pub joined: Option<f64>,
}

/// Starts this process's peak resident set again from its current resident
/// set (Linux `/proc/self/clear_refs` 5), so a block proved after another
/// reads its own peak, not the earlier block's levels': the base's memory
/// guard reads the peak. Best effort: `false` where it cannot.
pub(crate) fn reset_peak_rss() -> bool {
    std::fs::write("/proc/self/clear_refs", "5").is_ok()
}

/// `LAMBDA_VM_LEVELS_CARD_PAD_MB=<MiB>`: while the levels prove, the card
/// ledger promises at most its budget less this many MiB: the driver holds a
/// promise of that size from the base's end to its own (one the card never
/// allocates), so the levels' commits, openings and revives together stay
/// that far under the budget while the base keeps all of it. Unset or 0 (the
/// default): no pad. Read once a process; nonsense is refused.
pub const CARD_PAD_ENV: &str = "LAMBDA_VM_LEVELS_CARD_PAD_MB";

/// What [`CARD_PAD_ENV`] says, in bytes.
pub fn parse_card_pad(value: Option<&str>) -> Result<u64, String> {
    match value.map(str::trim) {
        None | Some("") => Ok(0),
        Some(v) => v
            .parse::<u64>()
            .map(|mb| mb << 20)
            .map_err(|_| format!("{CARD_PAD_ENV} must be a whole number of MiB, got `{v}`")),
    }
}

fn card_pad_bytes() -> Result<u64, String> {
    static PAD: std::sync::OnceLock<Result<u64, String>> = std::sync::OnceLock::new();
    PAD.get_or_init(|| parse_card_pad(std::env::var(CARD_PAD_ENV).ok().as_deref()))
        .clone()
}

/// The levels' pad, while the driver holds it, and its size.
static LEVELS_PAD: std::sync::Mutex<Option<(multilinear::gpu::DeviceRoom, u64)>> =
    std::sync::Mutex::new(None);

/// The bytes of the levels' pad held now (0 when none).
fn levels_pad_held() -> u64 {
    LEVELS_PAD
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .as_ref()
        .map_or(0, |(_, b)| *b)
}

/// Takes the levels' pad ([`CARD_PAD_ENV`]): the bytes held, 0 when there is
/// none to take or no card, or the ledger would not promise it.
fn take_levels_pad() -> Result<u64, String> {
    let bytes = card_pad_bytes()?;
    if bytes == 0 {
        return Ok(0);
    }
    let held = multilinear::gpu::reserve_room(bytes).map(|room| (room, bytes));
    let got = held.as_ref().map_or(0, |(_, b)| *b);
    *LEVELS_PAD.lock().unwrap_or_else(|e| e.into_inner()) = held;
    // Under the levels' split (`LAMBDA_VM_TWO_CARD_LEVELS`) the levels prove
    // on the second card too: the same pad there, promised in the second
    // role so it is that card's ledger's (and goes back to it when dropped).
    // A pad the second ledger refuses is not held there; the bytes returned
    // stay card 0's.
    if multilinear::gpu::two_card_levels_on() && multilinear::gpu::two_card_split_ready() {
        let second = multilinear::gpu::in_second_role(|| multilinear::gpu::reserve_room(bytes))
            .map(|room| (room, bytes));
        *SECOND_LEVELS_PAD.lock().unwrap_or_else(|e| e.into_inner()) = second;
    }
    Ok(got)
}

/// The levels' pad on the second card, under the levels' split, while the
/// driver holds it ([`take_levels_pad`]).
static SECOND_LEVELS_PAD: std::sync::Mutex<Option<(multilinear::gpu::DeviceRoom, u64)>> =
    std::sync::Mutex::new(None);

/// Gives the levels' pad back, on both cards.
fn release_levels_pad() {
    *LEVELS_PAD.lock().unwrap_or_else(|e| e.into_inner()) = None;
    *SECOND_LEVELS_PAD.lock().unwrap_or_else(|e| e.into_inner()) = None;
}

/// Gives the levels' pad back when the driver returns, by any path.
struct PadRelease;

impl Drop for PadRelease {
    fn drop(&mut self) {
        release_levels_pad();
    }
}

/// `LAMBDA_VM_LEVELS_CARD_BUDGET_PCT=<percent>`: from the base's end to the
/// driver's return, the card ledger's budget is this share of the card's
/// memory instead of the base's (80 %, or `LAMBDA_VM_VRAM_BUDGET_MB`); the
/// base's comes back when the driver returns, by any path. The base runs
/// beside its own pools, staging and caches; after it the driver settles the
/// card (every stream drained, the pool trimmed), and the levels' one table
/// of 2^29 cells on a large block needs its GKR trees on the card at once: at
/// 80 % its argue was refused a 1.73 GiB tree with 0.86 GiB of room and built
/// on the host (13.4 s). At 85 % the card's real peak through every argue is
/// read beside the ledger (`card at the argue`, phase B's card windows).
/// Unset or empty: [`LEVELS_CARD_BUDGET_DEFAULT_PCT`]. A whole number from 50
/// to 95 overrides it (80 gives the levels the base's default share);
/// anything else is refused. A prover's choice, never the proof's: a refused
/// promise has a host path.
pub const CARD_BUDGET_ENV: &str = "LAMBDA_VM_LEVELS_CARD_BUDGET_PCT";

/// The levels' share of the card when [`CARD_BUDGET_ENV`] says nothing.
pub const LEVELS_CARD_BUDGET_DEFAULT_PCT: u64 = 85;

/// What [`CARD_BUDGET_ENV`] says: the percent ([`LEVELS_CARD_BUDGET_DEFAULT_PCT`]
/// when unset or empty).
pub fn parse_card_budget(value: Option<&str>) -> Result<Option<u64>, String> {
    match value.map(str::trim) {
        None | Some("") => Ok(Some(LEVELS_CARD_BUDGET_DEFAULT_PCT)),
        Some(v) => match v.parse::<u64>() {
            Ok(pct) if (50..=95).contains(&pct) => Ok(Some(pct)),
            _ => Err(format!(
                "{CARD_BUDGET_ENV} must be a whole percent from 50 to 95, got `{v}`"
            )),
        },
    }
}

/// The levels' share of the card: [`CARD_BUDGET_ENV`], or its default; `None`
/// (the base's budget kept) when the knob says nothing and the process set
/// its whole card budget itself (`LAMBDA_VM_VRAM_BUDGET_MB`, a test's
/// throttle).
fn card_budget_pct() -> Result<Option<u64>, String> {
    static PCT: std::sync::OnceLock<Result<Option<u64>, String>> = std::sync::OnceLock::new();
    PCT.get_or_init(|| {
        let word = std::env::var(CARD_BUDGET_ENV).ok();
        let silent = word.as_deref().is_none_or(|w| w.trim().is_empty());
        if silent && std::env::var_os("LAMBDA_VM_VRAM_BUDGET_MB").is_some() {
            return Ok(None);
        }
        parse_card_budget(word.as_deref())
    })
    .clone()
}

/// The base's budget while the levels' own is in force ([`CARD_BUDGET_ENV`]).
static BASE_BUDGET: std::sync::Mutex<Option<u64>> = std::sync::Mutex::new(None);

/// Under the levels' split (`LAMBDA_VM_TWO_CARD_LEVELS`): the second card's
/// budget before the levels' was put in force there too.
static SECOND_BASE_BUDGET: std::sync::Mutex<Option<u64>> = std::sync::Mutex::new(None);

/// Puts the levels' budget in force ([`CARD_BUDGET_ENV`], 85 % by default) on
/// a card of `total` bytes: the budget now (unchanged without a card, or when
/// the process set its own card budget and the knob says nothing).
fn take_levels_budget(total: Option<u64>) -> Result<u64, String> {
    let (Some(pct), Some(total)) = (card_budget_pct()?, total) else {
        return Ok(multilinear::gpu::reserve_budget());
    };
    // The levels' split (`LAMBDA_VM_TWO_CARD_LEVELS`) argues and opens on the
    // second card as well: the same share there (the cards are alike), its own
    // budget back at the return; two backends on one card (forced, one card)
    // get half each, as at the second backend's start. Asked only with the
    // levels' knob on; ready makes the second backend if the prove's set-up
    // has not yet.
    let split = multilinear::gpu::two_card_levels_on() && multilinear::gpu::two_card_split_ready();
    let bytes = match split && multilinear::gpu::second_shares_card_0() {
        true => total / 100 * pct / 2,
        false => total / 100 * pct,
    };
    let base = multilinear::gpu::set_reserve_budget(bytes);
    let mut held = BASE_BUDGET.lock().unwrap_or_else(|e| e.into_inner());
    if held.is_none() {
        *held = Some(base);
    }
    if split && let Some(second) = multilinear::gpu::set_second_reserve_budget(bytes) {
        let mut held2 = SECOND_BASE_BUDGET.lock().unwrap_or_else(|e| e.into_inner());
        if held2.is_none() {
            *held2 = Some(second);
        }
        eprintln!(
            "LEVELS CARD budget on the second card: {:.2} GiB (LAMBDA_VM_TWO_CARD_LEVELS), its {:.2} GiB \
             back at the driver's return",
            bytes as f64 / (1u64 << 30) as f64,
            held2.unwrap_or(second) as f64 / (1u64 << 30) as f64
        );
    }
    let gib = |b: u64| b as f64 / (1u64 << 30) as f64;
    eprintln!(
        "LEVELS CARD budget: {:.2} GiB ({pct} % of {:.2}) for the levels, the base's {:.2} GiB back at \
         the driver's return",
        gib(bytes),
        gib(total),
        gib(held.unwrap_or(base))
    );
    Ok(bytes)
}

/// Puts the base's budget back, if the levels' was in force.
fn release_levels_budget() {
    if let Some(base) = BASE_BUDGET.lock().unwrap_or_else(|e| e.into_inner()).take() {
        multilinear::gpu::set_reserve_budget(base);
    }
    if let Some(second) = SECOND_BASE_BUDGET
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .take()
    {
        multilinear::gpu::set_second_reserve_budget(second);
    }
}

/// Puts the base's budget back when the driver returns, by any path.
struct BudgetRelease;

impl Drop for BudgetRelease {
    fn drop(&mut self) {
        release_levels_budget();
    }
}

/// `LAMBDA_VM_LEVELS_FREE_BESIDE=1`: level 1's filled program (its
/// instructions and instruction columns) is let go on the thread that hands
/// the execution's memory back beside the prove, where it is otherwise freed
/// on the driver's thread before the prove starts. The same program dropped,
/// later and elsewhere; the prove never reads it. Off by default (and for any
/// word but `1`). Read once a process.
pub const FREE_BESIDE_ENV: &str = "LAMBDA_VM_LEVELS_FREE_BESIDE";

/// A test's [`free_beside`], process wide (0 the knob, 1 off, 2 on).
static FREE_BESIDE_FOR_TEST: std::sync::atomic::AtomicU8 = std::sync::atomic::AtomicU8::new(0);

/// Tests only: [`free_beside`] forced for the process (`None` restores the
/// knob).
#[doc(hidden)]
pub fn set_free_beside_for_test(on: Option<bool>) {
    let v = match on {
        None => 0,
        Some(false) => 1,
        Some(true) => 2,
    };
    FREE_BESIDE_FOR_TEST.store(v, std::sync::atomic::Ordering::Relaxed);
}

/// Whether [`FREE_BESIDE_ENV`] is on (or a test's override).
pub fn free_beside() -> bool {
    match FREE_BESIDE_FOR_TEST.load(std::sync::atomic::Ordering::Relaxed) {
        1 => return false,
        2 => return true,
        _ => {}
    }
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        super::wlfm_knobs::knob_word_on(std::env::var(FREE_BESIDE_ENV).ok().as_deref())
    })
}

/// What [`take_program`] took out of a program: dropped wherever it goes.
pub(crate) struct TakenProgram {
    _instrs: Vec<super::instr::Instr>,
    _columns: Vec<Vec<crate::tables::types::FE>>,
}

/// A filled level's program let go: its instructions and instruction
/// columns (their counts kept in `sizes`). The prove reads the build and the
/// filled traces, never the program; the heights stay.
fn release_program(p: &mut LfmProgram, sizes: &mut Vec<(usize, usize)>) {
    drop(take_program(p, sizes));
}

/// Tests only: [`release_program`].
#[cfg(test)]
pub(crate) fn release_program_for_test(p: &mut LfmProgram, sizes: &mut Vec<(usize, usize)>) {
    release_program(p, sizes);
}

impl TakenProgram {
    /// Tests only: the instructions taken.
    #[cfg(test)]
    pub(crate) fn instrs_len(&self) -> usize {
        self._instrs.len()
    }
}

/// [`release_program`]'s taking without the freeing: the program left as
/// [`release_program`] leaves it, what it held handed back.
pub(crate) fn take_program(p: &mut LfmProgram, sizes: &mut Vec<(usize, usize)>) -> TakenProgram {
    sizes.push((p.instrs.len(), p.groups.b3l.real_rows));
    let instrs = std::mem::take(&mut p.instrs);
    let mut columns = Vec::with_capacity(15);
    let g = &mut p.groups;
    for group in [
        &mut g.const_,
        &mut g.balu,
        &mut g.xalu,
        &mut g.select,
        &mut g.bitdec,
        &mut g.hash,
        &mut g.keccak,
        &mut g.blake3,
        &mut g.lanes,
        &mut g.hint,
        &mut g.public,
        &mut g.b3l,
        &mut g.mux,
        &mut g.cap,
        &mut g.cap_read,
    ] {
        columns.push(std::mem::take(&mut group.data));
    }
    TakenProgram {
        _instrs: instrs,
        _columns: columns,
    }
}

/// The card at a point of the driver, after every stream's pending frees
/// landed and the memory pool handed back what it holds unused
/// (`math_cuda::device::drain_and_trim`): the pool's bytes in use and its
/// peak since the last reading, and the device's free and total bytes. A
/// pool holds freed blocks, and a block freed on one stream is not free to
/// another until that stream reaches the free: without this, a level's first
/// group commit can find the card full of what the stacks' commits or the
/// base let go. `None` without a card.
#[derive(Clone, Copy, Debug, Default)]
pub struct CardReading {
    pub pool_used: u64,
    pub pool_peak: u64,
    pub free: u64,
    pub total: u64,
}

/// [`CardReading`] now, the card drained and trimmed first, its pool's peak
/// started again.
pub(crate) fn card_settle(at: &str) -> Option<CardReading> {
    card_settle_on(at, true)
}

/// A card reading inside the levels: trimmed as today, or under
/// [`super::wlfm_knobs::levels_untrimmed`] untrimmed ([`card_settle_on`]).
fn card_settle_levels(at: &str) -> Option<CardReading> {
    card_settle_on(at, !super::wlfm_knobs::levels_untrimmed())
}

/// Card readings this process took without a trim ([`card_settle_on`]).
static UNTRIMMED: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// [`UNTRIMMED`] so far.
pub fn untrimmed_readings() -> u64 {
    UNTRIMMED.load(std::sync::atomic::Ordering::Relaxed)
}

/// [`card_settle`], trimming the pool only when `trim`: untrimmed, the pool
/// keeps the pages it holds for the next allocations.
fn card_settle_on(at: &str, trim: bool) -> Option<CardReading> {
    if !trim {
        UNTRIMMED.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    }
    let reading = if trim {
        multilinear::gpu::settle_card()
    } else {
        multilinear::gpu::settle_card_untrimmed()
    };
    // Under the stacks by card the second card takes a level's stacks: it is
    // drained (and read) at the same points, before its first stack.
    if stacks_by_card_acts() {
        second_card_settle(at, trim);
    }
    let (pool_used, pool_peak, free, total) = reading?;
    let gib = |b: u64| b as f64 / (1u64 << 30) as f64;
    // Said as it happens, so a run that fails later still shows it.
    eprintln!(
        "LEVELS CARD {at}: pool {:.2} GiB (peak before {:.2}) · free {:.2} of {:.2} GiB · ledger \
         {:.2} of {:.2} GiB promised",
        gib(pool_used),
        gib(pool_peak),
        gib(free),
        gib(total),
        gib(multilinear::gpu::ledger_reserved()),
        gib(multilinear::gpu::reserve_budget())
    );
    Some(CardReading {
        pool_used,
        pool_peak,
        free,
        total,
    })
}

/// [`card_settle_on`] on the second card, in its role: drained (trimmed when
/// `trim`) and read, printed as the second card's.
fn second_card_settle(at: &str, trim: bool) {
    multilinear::gpu::in_second_role(|| {
        let reading = if trim {
            multilinear::gpu::settle_card()
        } else {
            multilinear::gpu::settle_card_untrimmed()
        };
        if let Some((pool_used, pool_peak, free, total)) = reading {
            let gib = |b: u64| b as f64 / (1u64 << 30) as f64;
            eprintln!(
                "LEVELS CARD {at} (second card): pool {:.2} GiB (peak before {:.2}) · free {:.2} of \
                 {:.2} GiB · ledger {:.2} of {:.2} GiB promised",
                gib(pool_used),
                gib(pool_peak),
                gib(free),
                gib(total),
                gib(multilinear::gpu::ledger_reserved()),
                gib(multilinear::gpu::reserve_budget())
            );
        }
    });
}

/// `LAMBDA_VM_LEVELS_SETTLE_LATE=1`: at the base's end the driver reads the
/// card after draining it but WITHOUT trimming its pool, and commits level 1's
/// stacks at once on the pages the pool still holds; the trim comes at the
/// next reading, once level 1 is committed. Today's reading trims the base's
/// pool first (≈ 0.08–0.14 s on a real block, in level 1's tail). Every
/// allocation is the pool's and retries after a trim when it fails
/// (`math_cuda::device::alloc_or_trim`); the levels' budget reads only the
/// card's total. Off by default (and for any word but `1`). The order of the
/// work changes, never its bytes. Read once a process.
pub const SETTLE_LATE_ENV: &str = "LAMBDA_VM_LEVELS_SETTLE_LATE";

thread_local! {
    /// A test's [`settle_late`] on this thread.
    static SETTLE_LATE_FOR_TEST: std::cell::Cell<Option<bool>> = const { std::cell::Cell::new(None) };
}

/// Tests only: [`settle_late`] forced on this thread (`None` restores the
/// knob).
#[doc(hidden)]
pub fn set_settle_late_for_test(on: Option<bool>) {
    SETTLE_LATE_FOR_TEST.with(|c| c.set(on));
}

/// [`SETTLE_LATE_ENV`]'s word: on only for `1`.
pub fn settle_late_word(word: Option<&str>) -> bool {
    super::wlfm_knobs::knob_word_on(word)
}

/// Whether [`SETTLE_LATE_ENV`] is on (or this thread's test override).
pub fn settle_late() -> bool {
    if let Some(on) = SETTLE_LATE_FOR_TEST.with(std::cell::Cell::get) {
        return on;
    }
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| settle_late_word(super::wlfm_knobs::env_word(SETTLE_LATE_ENV).as_deref()))
}

/// The card read at the base's end, before level 1's stacks are committed:
/// trimmed as today, or under [`settle_late`] untrimmed. Its seconds and
/// whether it trimmed, for the times.
fn settle_base_end(late: bool) -> (Option<CardReading>, f64, bool) {
    let t = std::time::Instant::now();
    let reading = card_settle_on("base end", !late);
    (reading, t.elapsed().as_secs_f64(), !late)
}

/// This process's resident set, in bytes (Linux; `None` elsewhere).
pub(crate) fn resident_now() -> Option<u64> {
    let status = std::fs::read_to_string("/proc/self/status").ok()?;
    status
        .lines()
        .find_map(|l| l.strip_prefix("VmRSS:"))
        .and_then(|v| v.trim().trim_end_matches("kB").trim().parse::<u64>().ok())
        .map(|kib| kib << 10)
}

/// ★ The round form every level's chains take: the base's own
/// (`Some(shrink)` when the plan's block chains run the WHIR round form,
/// `None` under today's), so one format runs end to end. A function of the
/// plan, which a verifier derives from its own format: never read from a
/// proof.
pub fn level_rounds(plan: &WhirBlockPlan) -> Option<u8> {
    let (_, n_stack) = plan.group_stack(0);
    match plan.chain_shape(n_stack).form {
        multilinear::whir_chain::RoundForm::Fri => None,
        multilinear::whir_chain::RoundForm::Whir(w) => u8::try_from(w.shrink()).ok(),
    }
}

/// One level built from its program at [`LEVEL_RATE`], its chains under
/// `rounds` ([`level_rounds`]), its prepared stacks committed on the host
/// when `on_host`: level `at.1` of a chain stopping at `at.0`, under every
/// level's levers, or `last`'s when it is the last ([`prepare_as_level`]).
pub(crate) fn build_level(
    program: LfmProgram,
    family: WlfmFamily,
    rounds: Option<u8>,
    on_host: bool,
    at: (LevelStop, usize),
    last: LastLevel,
) -> Result<Level, String> {
    let hasher = program.hasher(<BlockFamily as ProofFamily>::SOCKET);
    let place = if on_host {
        StackPlace::Host
    } else {
        StackPlace::Card
    };
    let build = prepare_as_level(at.0, at.1, last, family, |levers| {
        prepare_wlfm_rounds(&program, hasher, LEVEL_RATE, levers, rounds)
    })?
    .commit(place)?;
    Ok(Level { program, build })
}

/// Level `k + 1`'s view of level `k`: its artifacts and the top's word count.
/// The most pieces a streamed execution lands (one bit a piece).
const MAX_STREAM_PIECES: usize = 64;

/// Level 2 beside level 1's prove ([`stream_next`]): its program and prebuild
/// from `prepared` (joined here), and its execution streamed over the pieces
/// `pieces` hands on ([`stream_level_over_pieces_parts`]); `None` for the execution
/// when its arena has more pieces than a streamed execution lands (level 2
/// then executes after level 1's prove, as without the knob).
///
/// With `early` ([`commit_next_early`]), the prebuild is handed on there the
/// moment it is joined, before the execution streams, for the driver to
/// commit as soon as level 1's prove returns; it comes back here only if the
/// driver has stopped taking it. Its fill runs its parts at once under
/// `fill_parts` ([`fill_parts`]).
fn stream_next_level(
    prepared: std::thread::ScopedJoinHandle<'_, Result<PreparedNext, String>>,
    pieces: mpsc::Receiver<(usize, Vec<LfmWord>)>,
    t0: std::time::Instant,
    opts: StreamOpts,
    early: Option<mpsc::SyncSender<WlfmPrebuild>>,
) -> Result<StreamedNext, String> {
    let (program, pre, marks) = prepared
        .join()
        .map_err(|_| "level 2's preparation panicked".to_string())??;
    let pre = match early {
        Some(tx) => tx.send(pre).err().map(|mpsc::SendError(pre)| pre),
        None => Some(pre),
    };
    if marks.len() > MAX_STREAM_PIECES {
        return Ok((program, pre, None));
    }
    let hasher = program.hasher(<BlockFamily as ProofFamily>::SOCKET);
    let streamed = stream_level_over_pieces_parts(&program, &marks, hasher, pieces, t0, opts)?;
    Ok((program, pre, Some(streamed)))
}

/// [`stream_next_level`] with level 2's preparation beside its execution
/// ([`next_beside`]): the program is made here, then the prebuild on a thread
/// of its own while this one streams the execution ([`prepare_and_stream`]).
/// The same program, prebuild and execution. Level `index` is prepared under
/// the levers it proves at ([`prepare_as_level`]), as [`prepare_next`] does.
#[allow(clippy::too_many_arguments)]
fn stream_next_level_beside(
    child: WlfmChild,
    family: WlfmFamily,
    stop: LevelStop,
    index: usize,
    last: LastLevel,
    pieces: mpsc::Receiver<(usize, Vec<LfmWord>)>,
    t0: std::time::Instant,
    opts: StreamOpts,
    early: Option<mpsc::SyncSender<WlfmPrebuild>>,
) -> Result<StreamedNext, String> {
    let (program, marks) = next_program(&child)?;
    let rounds = child.artifacts.rounds;
    let prepare = || {
        prepare_as_level(stop, index, last, family, |levers| {
            prepare_level_as(&program, levers, rounds)
        })
    };
    let (pre, streamed) = prepare_and_stream(&program, &marks, prepare, pieces, t0, opts, early)?;
    Ok((program, pre, streamed))
}

/// [`prepare_and_stream`]'s level: its prebuild (`None` when handed on) and
/// its execution streamed and filled (`None` when it does not stream).
type PreparedBeside = (
    Option<WlfmPrebuild>,
    Option<(super::proof::LfmFilled, NextStream)>,
);

/// `program`'s prebuild (`prepare`) on a scoped thread of its own,
/// handed on to `early` the moment it exists (it comes back only if `early`
/// has stopped taking it), beside its execution streamed over `pieces` on
/// this thread ([`stream_level_over_pieces_parts`]; none when its arena has
/// more pieces than a streamed execution lands). A failed or panicking
/// preparation is refused after the execution has ended; its sender is
/// dropped as it ends, so a receiver on `early` never waits for it.
pub(super) fn prepare_and_stream(
    program: &LfmProgram,
    marks: &[u32],
    prepare: impl FnOnce() -> Result<WlfmPrebuild, String> + Send,
    pieces: mpsc::Receiver<(usize, Vec<LfmWord>)>,
    t0: std::time::Instant,
    opts: StreamOpts,
    early: Option<mpsc::SyncSender<WlfmPrebuild>>,
) -> Result<PreparedBeside, String> {
    let hasher = program.hasher(<BlockFamily as ProofFamily>::SOCKET);
    std::thread::scope(|inner| {
        let prepare = inner.spawn(move || -> Result<(Option<WlfmPrebuild>, f64), String> {
            let pre = prepare()?;
            let at = t0.elapsed().as_secs_f64();
            let pre = match early {
                Some(tx) => tx.send(pre).err().map(|mpsc::SendError(pre)| pre),
                None => Some(pre),
            };
            Ok((pre, at))
        });
        let streamed = (marks.len() <= MAX_STREAM_PIECES)
            .then(|| stream_level_over_pieces_parts(program, marks, hasher, pieces, t0, opts));
        let (pre, at) = prepare
            .join()
            .map_err(|_| "level 2's preparation panicked".to_string())??;
        let streamed = streamed.transpose()?.map(|(filled, mut times)| {
            times.prebuilt = Some(at);
            (filled, times)
        });
        Ok((pre, streamed))
    })
}

/// `program`'s execution over one arena cut at `marks` (each piece's end:
/// the front is piece 0, a child's group `g` is piece `g + 1`), each piece
/// landed as `pieces` hands it on, in any order, and filled once the last has
/// landed: [`super::wlfm::execute_and_fill_unmerged`]'s traces over the
/// pieces joined. The channel closing before every piece came refuses.
#[cfg(test)]
pub(super) fn stream_level_over_pieces(
    program: &LfmProgram,
    marks: &[u32],
    hasher: super::HasherKind,
    pieces: mpsc::Receiver<(usize, Vec<LfmWord>)>,
    t0: std::time::Instant,
) -> Result<(super::proof::LfmFilled, NextStream), String> {
    stream_level_over_pieces_parts(program, marks, hasher, pieces, t0, StreamOpts::default())
}

/// [`stream_level_over_pieces`] with its fill's parts at once and its waves
/// in program order as `opts` says ([`fill_parts`], [`waves_in_order`]).
fn stream_level_over_pieces_parts(
    program: &LfmProgram,
    marks: &[u32],
    hasher: super::HasherKind,
    pieces: mpsc::Receiver<(usize, Vec<LfmWord>)>,
    t0: std::time::Instant,
    opts: StreamOpts,
) -> Result<(super::proof::LfmFilled, NextStream), String> {
    use super::executor::{ArenaPiece, StreamedExecution};
    let now = || t0.elapsed().as_secs_f64();
    if marks.is_empty() || marks.len() > MAX_STREAM_PIECES {
        return Err(format!("{} pieces do not stream", marks.len()));
    }
    let from = now();
    let mut start = 0u32;
    let mut arena_pieces = Vec::with_capacity(marks.len());
    for (group, &end) in marks.iter().enumerate() {
        let len = end
            .checked_sub(start)
            .ok_or("the arena's marks go backwards")?;
        arena_pieces.push(ArenaPiece {
            arena: 0,
            start,
            len,
            group,
        });
        start = end;
    }
    let npieces = arena_pieces.len();
    let held: Vec<std::cell::OnceCell<Vec<LfmWord>>> =
        (0..npieces).map(|_| std::cell::OnceCell::new()).collect();
    let mut times = NextStream {
        pieces: npieces,
        from,
        ..NextStream::default()
    };
    let execution = {
        let mut ex = StreamedExecution::new_pieced_in_order(
            program,
            arena_pieces,
            &hasher,
            false,
            bucket_waves(),
            setup_fast(),
            opts.in_order,
        )
        .map_err(|e| format!("the streamed execution: {e:?}"))?;
        let mut landed = 0usize;
        for (i, words) in pieces {
            let slot = held.get(i).ok_or("a piece the arena does not have")?;
            slot.set(words)
                .map_err(|_| "a piece landed twice".to_string())?;
            let words = slot.get().ok_or("the piece")?;
            let want = marks[i] - if i == 0 { 0 } else { marks[i - 1] };
            if words.len() != want as usize {
                return Err(format!(
                    "piece {i} has {} words, its mark {want}",
                    words.len()
                ));
            }
            ex.land_piece(i, words)
                .map_err(|e| format!("piece {i}: {e:?}"))?;
            landed += 1;
            if landed == 1 {
                times.first = now();
            }
            times.last = now();
            if landed == npieces {
                break;
            }
        }
        if landed != npieces {
            return Err(format!("the pieces stopped at {landed} of {npieces}"));
        }
        ex.finish().map_err(|e| format!("the execution: {e:?}"))?
    };
    times.executed = now();
    let t = std::time::Instant::now();
    let filled = super::proof::lfm_fill_executed_parts(
        program,
        execution,
        hasher,
        times.executed - from,
        &Vec::new(),
        super::wlfm::chip_records(),
        None,
        opts.fill_parts,
    );
    times.fill = t.elapsed().as_secs_f64();
    Ok((filled, times))
}

pub fn child_of(plan: &WhirBlockPlan, level: &Level) -> WlfmChild {
    child_of_build(plan, &level.build)
}

fn child_of_build(plan: &WhirBlockPlan, build: &WlfmBuild) -> WlfmChild {
    WlfmChild {
        artifacts: build.artifacts.clone(),
        num_public_words: plan.top_layout().total(),
        cap_mux: plan.cap_mux(),
    }
}

/// Every level of `plan`'s chain under the default stop, the last level
/// under the process's last-level format: the tests' derivation (the
/// verifier derives through [`derive_levels_last`] under its named format).
/// `on_host = false` commits the prepared stacks on the card. `plan` must be
/// one leaf over every group.
#[cfg(test)]
pub(crate) fn derive_levels(
    plan: &WhirBlockPlan,
    family: WlfmFamily,
    on_host: bool,
) -> Result<Vec<Level>, String> {
    derive_levels_to(plan, family, LevelStop::DEFAULT, on_host)
}

/// [`derive_levels`] under the chain stop `stop`, the last level under the
/// process's last-level format ([`super::last_level::last_level`]); test
/// only.
#[cfg(test)]
pub(crate) fn derive_levels_to(
    plan: &WhirBlockPlan,
    family: WlfmFamily,
    stop: LevelStop,
    on_host: bool,
) -> Result<Vec<Level>, String> {
    let last = super::last_level::last_level(family)?;
    derive_levels_last(plan, family, stop, last, on_host)
}

/// ★ Every level of `plan`'s chain, its programs emitted and built, the last
/// level under the NAMED format `last`: ONE derivation for the prover
/// (`on_host = false`: prepared commits on the card) and the verifier
/// (`on_host = true`). It reads no knob. `plan` must be one leaf over every
/// group.
pub fn derive_levels_last(
    plan: &WhirBlockPlan,
    family: WlfmFamily,
    stop: LevelStop,
    last: LastLevel,
    on_host: bool,
) -> Result<Vec<Level>, String> {
    check_family(plan, family)?;
    let more = |levels: &[Level]| -> Result<bool, String> {
        let groups = levels.last().expect("a level").build.artifacts.sizes.len();
        match stop {
            LevelStop::Levels(0) => Err("a chain has at least one level".into()),
            LevelStop::Levels(n) => Ok(levels.len() < n.min(MAX_LEVELS)),
            LevelStop::Groups(g) if groups <= g => Ok(false),
            LevelStop::Groups(_) if levels.len() == MAX_LEVELS => Err(format!(
                "{MAX_LEVELS} levels and the last still has {groups} groups"
            )),
            LevelStop::Groups(_) => Ok(true),
        }
    };
    if let LevelStop::Levels(0) = stop {
        return Err("a chain has at least one level".into());
    }
    let rounds = level_rounds(plan);
    let mut levels = vec![build_level(
        plan.closed_block_program()?,
        family,
        rounds,
        on_host,
        (stop, 1),
        last,
    )?];
    while more(&levels)? {
        let child = child_of(plan, levels.last().expect("a level"));
        let rounds = child.artifacts.rounds;
        let index = levels.len() + 1;
        levels.push(build_level(
            wlfm_leg_program(&child)?,
            family,
            rounds,
            on_host,
            (stop, index),
            last,
        )?);
    }
    Ok(levels)
}

/// One level's seconds, by part, as its prover measured them.
#[derive(Clone, Copy, Debug, Default)]
pub struct LevelSecs {
    pub arena: f64,
    pub prove: f64,
}

/// ★ Proves every level over `base`: level 1 over the base proof's witness,
/// each next level over the previous level's proof. Returns every level's
/// proof, the last first-class (the one a verifier checks).
pub fn prove_levels<L: crate::block_whir::LocalPart>(
    plan: &WhirBlockPlan,
    levels: &[Level],
    base: &BlockWhirProof<L>,
) -> Result<(Vec<WlfmProof>, Vec<LevelSecs>), String> {
    let mut proofs: Vec<WlfmProof> = Vec::with_capacity(levels.len());
    let mut secs = Vec::with_capacity(levels.len());
    for (k, level) in levels.iter().enumerate() {
        let t = std::time::Instant::now();
        let arenas = if k == 0 {
            block_leaf_arena(plan, base, 0)?
        } else {
            let child = child_of(plan, &levels[k - 1]);
            vec![wlfm_leg_arena(&child, &proofs[k - 1])?]
        };
        let arena = t.elapsed().as_secs_f64();
        let t = std::time::Instant::now();
        let filled = super::wlfm::execute_and_fill_unmerged(
            &level.program,
            &arenas,
            level.build.artifacts.hasher,
        )?;
        let proof =
            prove_wlfm_traces_as(&level.build, filled, prefix_check(), WlfmProving::Streamed)
                .map_err(|e| format!("level {}: {e}", k + 1))?;
        secs.push(LevelSecs {
            arena,
            prove: t.elapsed().as_secs_f64(),
        });
        proofs.push(proof);
    }
    Ok((proofs, secs))
}

/// ★ The block's public verifier over its last level, at the pinned format:
/// [`verify_block_levels_at`] with every named choice at its pinned value —
/// [`BlockFormat::pinned_with`](crate::block_whir::BlockFormat::pinned_with)
/// `Indexed`'s blowup, leaf order, round-0 trees, leaf scale, leaf query, GKR
/// packing, memory format and page runs; the BLAKE3 base's FRI round form;
/// the packed cap check — the default chain stop and the family's pinned
/// last level ([`LastLevel::pinned`]: every lever on under BLAKE3, every
/// lever off under Poseidon1).
pub fn verify_block_levels(
    elf: &[u8],
    statement: crate::block_whir::BlockStatement<'_>,
    family: WlfmFamily,
    last: &WlfmProof,
) -> Result<(), String> {
    let format = pin_family(
        crate::block_whir::BlockFormat::pinned_with(GroupFold::Indexed),
        family,
        crate::block_whir::B3Rounds::Fri,
    )?;
    verify_block_levels_pinned(
        elf,
        &format,
        statement,
        family,
        LevelStop::DEFAULT,
        LastLevel::pinned(family),
        CapMux::Packed,
        last,
    )
}

/// ★ [`verify_block_levels`] over the last level's FILE
/// ([`super::last_level::last_level_file`]): the file is read for the
/// family's pinned last level ([`LastLevel::pinned`]), so a file of another
/// version or made in another last-level format is refused by name before
/// anything is decoded ([`super::last_level::read_last_level_file`]), and the
/// proof it holds is verified in full by [`verify_block_levels`].
pub fn verify_block_levels_file(
    elf: &[u8],
    statement: crate::block_whir::BlockStatement<'_>,
    family: WlfmFamily,
    file: &[u8],
) -> Result<(), String> {
    let last = super::last_level::read_last_level_file(file, LastLevel::pinned(family))?;
    verify_block_levels(elf, statement, family, &last)
}

/// ★ The block's public verifier over its last level, at the pinned format
/// with the caller's NAMED choices (as the block tree's
/// `whir_block::verify_block_tree_at` takes them), plus the BLAKE3 base's
/// round form, the cap check, the chain stop and the last level's own format
/// (`last_level`, [`super::last_level`]). Every one is the verifier's
/// constant: the proof carries none of them, the format's tag is in the
/// statement, and a caller cannot hand in a format, proof options or a plan.
/// Then level 1's plan is derived here ([`verify_block_levels_pinned`]): its
/// prepared roots committed on the host, the base's options under the
/// format's own WHIR format, the leaves' share form `p · (1/q)` — never
/// `LFM_WHIR_SHARE_INVERSE` — and the named cap check — never
/// `LAMBDA_VM_LFM_CAP_MUX`, named before the derivation emits a program. The
/// format knobs the derivation still reads (the BLAKE3 base's cap and final
/// polynomial, the `LFM_HASH` split) are refused off their pinned values
/// ([`check_process_format`]).
#[allow(clippy::too_many_arguments)]
pub fn verify_block_levels_at(
    elf: &[u8],
    statement: crate::block_whir::BlockStatement<'_>,
    family: WlfmFamily,
    stop: LevelStop,
    last_level: LastLevel,
    last: &WlfmProof,
    group: GroupFold,
    blowup: crate::zf_format::WhirBlowup,
    leaves: multilinear::whir_chain::LeafOrder,
    trees: multilinear::whir_chain::Round0Trees,
    leaf_scale: super::whir_block::LeafScale,
    leaf_query: super::whir_chain::LeafQuery,
    pack: multilinear::whir_chain::GkrPack,
    memory: crate::tables::fused_memory::MemoryFormat,
    page_run_log2: u8,
    rounds: crate::block_whir::B3Rounds,
    cap_mux: CapMux,
) -> Result<(), String> {
    let presets = super::whir_block::verifier_presets_at(
        group,
        blowup,
        leaves,
        trees,
        leaf_scale,
        leaf_query,
        pack,
        memory,
        page_run_log2,
    )?;
    let format = pin_family(presets.format, family, rounds)?;
    verify_block_levels_pinned(
        elf, &format, statement, family, stop, last_level, cap_mux, last,
    )
}

/// A pinned block format moved onto `family`'s base: unchanged for
/// [`WlfmFamily::P1`] (whose chains take no round form), and for
/// [`WlfmFamily::B3`] the 7-round BLAKE3 base with its own constants — its
/// group size ([`crate::block_whir::B3_BASE_GROUP_POLYS`]), its GKR ladder
/// ([`crate::block_whir::B3_BASE_GKR_DEFAULT`]), KECCAK_RND's form (its byte
/// lookups argued per group, [`crate::block_whir::B3_BASE_KECCAK_RND`]) and
/// the round form `rounds` — never their environment knobs. The leaf order and round-0 trees are
/// re-checked against the new base.
pub(crate) fn pin_family(
    mut format: crate::block_whir::BlockFormat,
    family: WlfmFamily,
    rounds: crate::block_whir::B3Rounds,
) -> Result<crate::block_whir::BlockFormat, String> {
    match family {
        WlfmFamily::P1 => {
            if rounds != crate::block_whir::B3Rounds::Fri {
                return Err(format!(
                    "the round form {rounds:?} is the BLAKE3 base's alone"
                ));
            }
        }
        WlfmFamily::B3 => {
            if let crate::block_whir::B3Rounds::Whir { shrink, .. } = rounds
                && !(1..=3).contains(&shrink)
            {
                return Err(format!("the round form's shrink is 1 to 3, not {shrink}"));
            }
            format.zf.base = BaseFormat::B3_WHIR;
            format.group_polys = crate::block_whir::B3_BASE_GROUP_POLYS;
            format.b3_rounds = rounds;
            format.argue = format
                .argue
                .with_ladder(crate::block_whir::B3_BASE_GKR_DEFAULT);
            format.keccak_rnd = crate::block_whir::B3_BASE_KECCAK_RND;
        }
    }
    format.zf.check_leaves()?;
    format.zf.check_trees()?;
    check_base(&format.zf.base, family)?;
    Ok(format)
}

/// The levels verifier's plan under `format`: level 1's plan with its
/// prepared roots committed on the host (`None`), the base's options under
/// `format.zf` (never the process's), the share form `p · (1/q)` and the cap
/// check `cap_mux`, named before any program is emitted. Neither the share
/// form nor the cap check is read from the environment; the knobs that
/// remain are refused by [`check_process_format`] in the public entries.
pub(crate) fn verifier_level_plan(
    elf: &[u8],
    format: &crate::block_whir::BlockFormat,
    statement: crate::block_whir::BlockStatement<'_>,
    family: WlfmFamily,
    cap_mux: CapMux,
) -> Result<WhirBlockPlan, String> {
    let options = super::proof::block_base_options_under(&format.zf);
    verifier_level_plan_with(elf, &options, format, statement, family, cap_mux)
}

/// Tests only: a BLAKE3 verifier's level plan under the fixture's proof
/// options and the process's cap check ([`verifier_level_plan_with`]).
#[cfg(test)]
pub(crate) fn verifier_level_plan_for_test(
    elf: &[u8],
    proof_options: &crate::ProofOptions,
    format: &crate::block_whir::BlockFormat,
    statement: crate::block_whir::BlockStatement<'_>,
) -> Result<WhirBlockPlan, String> {
    verifier_level_plan_with(
        elf,
        proof_options,
        format,
        statement,
        WlfmFamily::B3,
        super::merkle_cap::cap_mux(),
    )
}

/// [`verifier_level_plan`] under the caller's proof options: the fixture
/// options a test proves its base under.
fn verifier_level_plan_with(
    elf: &[u8],
    proof_options: &crate::ProofOptions,
    format: &crate::block_whir::BlockFormat,
    statement: crate::block_whir::BlockStatement<'_>,
    family: WlfmFamily,
    cap_mux: CapMux,
) -> Result<WhirBlockPlan, String> {
    level_plan_with(
        elf,
        proof_options,
        format,
        statement,
        family,
        None,
        true,
        Some(cap_mux),
        true,
    )
}

/// The verifier body under a pinned `format`: [`verifier_level_plan`], then
/// [`verify_levels_to`] with the last level under the named `last_level`.
#[allow(clippy::too_many_arguments)]
fn verify_block_levels_pinned(
    elf: &[u8],
    format: &crate::block_whir::BlockFormat,
    statement: crate::block_whir::BlockStatement<'_>,
    family: WlfmFamily,
    stop: LevelStop,
    last_level: LastLevel,
    cap_mux: CapMux,
    last: &WlfmProof,
) -> Result<(), String> {
    check_process_format(family, last_level)?;
    check_last_level_family(last_level, family)?;
    let plan = verifier_level_plan(elf, format, statement, family, cap_mux)?;
    verify_levels_to(&plan, family, stop, last_level, last)
}

/// One binary, one format: the public entries refuse a process whose knobs
/// would select another honest configuration than the pinned one —
/// under the BLAKE3 base its cap height and final polynomial
/// (`LAMBDA_VM_B3_BASE_CAP`, `LAMBDA_VM_B3_BASE_FINAL`, which the base's
/// chains and its statement tag read) and its local lookups
/// (`LAMBDA_VM_B3_BASE_LOCAL` off, which the prover's format reads; a word
/// asking for them is refused under Poseidon1, whose format aborts on it),
/// and under the Poseidon1 base the
/// `LFM_HASH` split (`LAMBDA_VM_LFM_HASH_SPLIT`, which every level program's
/// identity reads). Under BLAKE3 the split is inert: a lean program has no
/// hash row (pinned by `lean_mux_tests::the_blake3_w_leg_program_is_pinned`
/// and `levels_tests::the_public_levels_verifier_accepts_at_the_pinned_format`).
///
/// The last level's own format (`LAMBDA_VM_LAST_LEVEL_EVALUATED`,
/// `LAMBDA_VM_LAST_LEVEL_FIRST_FOLD`, `LAMBDA_VM_LAST_LEVEL_FINAL`,
/// [`super::last_level`]) must be the verifier's named one (`last_level`;
/// the family's pinned one, [`LastLevel::pinned`], at
/// [`verify_block_levels`]): a process whose knobs resolve to another is
/// refused ([`check_last_level_knobs`]).
///
/// Each knob's word is parsed first, from the environment, by the knob's own
/// refusing parser: a malformed word, under either family, is an `Err` here,
/// never the abort or panic of the process-wide reader. Only then are the
/// process-wide cells read — the values the derivation will use — and only
/// the family's own: a Poseidon1 verification reads no BLAKE3 cell.
fn check_process_format(family: WlfmFamily, last_level: LastLevel) -> Result<(), String> {
    check_last_level_knobs(super::last_level::last_level(family)?, last_level)?;
    let cap = knob_word(
        crate::block_whir::B3_BASE_CAP_ENV,
        crate::block_whir::b3_base_cap_setting,
    )?;
    let final_poly = knob_word(
        crate::block_whir::B3_BASE_FINAL_ENV,
        crate::block_whir::b3_base_final_setting,
    )?;
    let hash_split = knob_word(super::chunking::HASH_SPLIT_ENV, hash_split_word)?;
    let local = knob_word(
        crate::block_whir::B3_BASE_LOCAL_ENV,
        crate::block_whir::b3_base_local_setting,
    )?;
    let (cap, final_poly, hash_split, local) = match family {
        WlfmFamily::B3 => (
            crate::block_whir::b3_base_cap(),
            crate::block_whir::b3_base_final(),
            hash_split,
            crate::block_whir::b3_base_local(),
        ),
        WlfmFamily::P1 => (
            cap,
            final_poly,
            super::chunking::hash_split_enabled(),
            local,
        ),
    };
    check_format_knobs(family, cap, final_poly, hash_split, local)
}

/// A knob's word in this process, through its own refusing parser: unset is
/// the parser's `None`, and a word that is not Unicode is refused by name.
fn knob_word<T>(name: &str, parse: fn(Option<&str>) -> Result<T, String>) -> Result<T, String> {
    match std::env::var(name) {
        Ok(word) => parse(Some(&word)),
        Err(std::env::VarError::NotPresent) => parse(None),
        Err(std::env::VarError::NotUnicode(word)) => Err(format!("{name}={word:?}: not a word")),
    }
}

/// `LAMBDA_VM_LFM_HASH_SPLIT`'s word as `chunking::hash_split_setting` reads
/// it, refused rather than panicked on when it is not `0` or `1`.
fn hash_split_word(raw: Option<&str>) -> Result<bool, String> {
    match raw.map(str::trim) {
        None | Some("") => Ok(super::chunking::HASH_SPLIT_DEFAULT),
        Some("0") => Ok(false),
        Some("1") => Ok(true),
        Some(other) => Err(format!(
            "{} must be 0 or 1, got {other:?}",
            super::chunking::HASH_SPLIT_ENV
        )),
    }
}

/// [`check_process_format`] at the family's pinned last level (the one
/// [`verify_block_levels`] names), for the tests that run it in a process of
/// their own.
#[cfg(test)]
pub(crate) fn check_process_format_for_tests(family: WlfmFamily) -> Result<(), String> {
    check_process_format(family, LastLevel::pinned(family))
}

/// One binary, one format, for the last level: the process's last-level
/// knobs (`process`, [`super::last_level::last_level`]) must name the
/// verifier's own `named` format. The derivation takes `named`, never the
/// process's; a process that would prove another last level is refused.
pub(crate) fn check_last_level_knobs(process: LastLevel, named: LastLevel) -> Result<(), String> {
    if process != named {
        return Err(format!(
            "the process asks for the last level {process:?}: the levels verifier checks \
             {named:?}"
        ));
    }
    Ok(())
}

/// [`check_process_format`] on given values; `local` is the local-lookups
/// knob's word (`None` unset).
pub(crate) fn check_format_knobs(
    family: WlfmFamily,
    cap: u8,
    final_poly: multilinear::whir_chain::FinalPoly,
    hash_split: bool,
    local: Option<bool>,
) -> Result<(), String> {
    match family {
        WlfmFamily::B3 => {
            if local.unwrap_or(crate::block_whir::B3_BASE_LOCAL_DEFAULT)
                != crate::block_whir::B3_BASE_LOCAL_DEFAULT
            {
                return Err(format!(
                    "{} turns local lookups off: the levels verifier checks the pinned \
                     KECCAK_RND form, its byte lookups argued per group",
                    crate::block_whir::B3_BASE_LOCAL_ENV
                ));
            }
            let pinned_cap = crate::block_whir::b3_base_cap_setting(None)?;
            if cap != pinned_cap {
                return Err(format!(
                    "{} asks for a cap of height {cap}: the levels verifier checks the pinned \
                     {pinned_cap}",
                    crate::block_whir::B3_BASE_CAP_ENV
                ));
            }
            let pinned_final = crate::block_whir::b3_base_final_setting(None)?;
            if final_poly != pinned_final {
                return Err(format!(
                    "{} asks for {final_poly:?}: the levels verifier checks the pinned \
                     {pinned_final:?}",
                    crate::block_whir::B3_BASE_FINAL_ENV
                ));
            }
        }
        WlfmFamily::P1 => {
            if local == Some(true) {
                return Err(format!(
                    "{} asks for local lookups: they are the BLAKE3 base's alone",
                    crate::block_whir::B3_BASE_LOCAL_ENV
                ));
            }
            if hash_split != super::chunking::HASH_SPLIT_DEFAULT {
                return Err(format!(
                    "{} asks for another LFM_HASH split: the levels verifier checks the \
                     pinned one",
                    super::chunking::HASH_SPLIT_ENV
                ));
            }
        }
    }
    Ok(())
}

/// [`verify_block_levels`] under a format and proof options of the caller's:
/// the fixture formats and options the tests prove at, and the process's cap
/// check. The plan is still derived here (prepared roots on the host, the
/// inverse share form). Test-only.
#[cfg(test)]
pub(crate) fn verify_block_levels_under(
    elf: &[u8],
    proof_options: &crate::ProofOptions,
    format: &crate::block_whir::BlockFormat,
    statement: crate::block_whir::BlockStatement<'_>,
    family: WlfmFamily,
    stop: LevelStop,
    last: &WlfmProof,
) -> Result<(), String> {
    let plan = verifier_level_plan_with(
        elf,
        proof_options,
        format,
        statement,
        family,
        super::merkle_cap::cap_mux(),
    )?;
    verify_levels_to(
        &plan,
        family,
        stop,
        super::last_level::last_level(family)?,
        last,
    )
}

/// The block's verifier over its last level, against a plan the caller
/// derived: every level derived on the host from `plan` ([`derive_levels`]),
/// `last` verified against the last level's artifacts, and its words claiming
/// the block. It trusts `plan`, its prepared roots included, so it is not a
/// public entry: [`verify_block_levels`] derives the plan itself.
#[cfg(test)]
pub(crate) fn verify_levels(
    plan: &WhirBlockPlan,
    family: WlfmFamily,
    last: &WlfmProof,
) -> Result<(), String> {
    verify_levels_to(
        plan,
        family,
        LevelStop::DEFAULT,
        super::last_level::last_level(family)?,
        last,
    )
}

/// [`verify_levels`] under the chain stop `stop` and the last level's format
/// `last_level` (the verifier's constants).
pub(crate) fn verify_levels_to(
    plan: &WhirBlockPlan,
    family: WlfmFamily,
    stop: LevelStop,
    last_level: LastLevel,
    last: &WlfmProof,
) -> Result<(), String> {
    let levels = derive_levels_last(plan, family, stop, last_level, true)?;
    let top = levels.last().expect("a level");
    verify_wlfm(&top.build.artifacts, last, &last.public_words).map_err(|e| {
        format!(
            "the last level does not verify (this verifier checks the last level's format: \
             {last_level}; a proof made in another is refused): {e}"
        )
    })?;
    if !top_claims(plan, &last.public_words) {
        return Err("the last level does not claim this block".to_string());
    }
    Ok(())
}

// ============================ the block's driver ===========================

/// The seconds of one level, on the driver's clock (seconds since it
/// started).
#[derive(Clone, Copy, Debug, Default)]
pub struct LevelClock {
    /// When its execute could start, and when it ended (level 1: the last
    /// streamed wave, after the base's end).
    pub execute_from: f64,
    pub executed: f64,
    /// Its fill's seconds and its prove's (the W-LFM prove, card held).
    pub fill: f64,
    pub prove: f64,
    pub end: f64,
}

/// What a levels run measured.
#[derive(Clone, Debug, Default)]
pub struct LevelsTimes {
    /// Under [`STATEMENT_EARLY_ENV`], when the base's early statement came
    /// and what level 1 made of it ([`EarlyNote`]); `None` when none came.
    pub early: Option<EarlyNote>,
    /// Under `LAMBDA_VM_LEVELS_BUILDER_WHOLE_POOL`, what level 1's builder
    /// did once the base returned ([`super::wlfm::WholePoolNote`]).
    pub builder_whole_pool: Option<super::wlfm::WholePoolNote>,
    /// Whether level 1's validator was joined after its fill
    /// ([`VALIDATE_AFTER_FILL_ENV`]), and the seconds that join waited.
    pub validated_after_fill: Option<f64>,
    /// Level 1's program let go on the driver's thread before its prove: the
    /// seconds there, and whether it went to the purge's thread instead
    /// ([`FREE_BESIDE_ENV`]).
    pub release: (f64, bool),
    /// The base proof's end, the plan's derivation, level 1's program emitted
    /// and every level built.
    pub base: f64,
    pub planned: f64,
    pub emitted: f64,
    pub built: f64,
    /// Level 1's streamed execution: its pieces, the waves landed before the
    /// base ended and their seconds, the seconds after it.
    pub pieces: usize,
    pub waves_before_base: usize,
    pub wave_secs_before_base: f64,
    /// How level 1's pieces landed ([`Level1Landing`]).
    pub landing: Level1Landing,
    /// Level 1's execution split ([`super::executor::ExecSplit`]): its
    /// schedule, setup, hashing (on the pool and on the driver's thread),
    /// applying and the rest. `None` before level 1 executes.
    pub level1_exec: Option<super::executor::ExecSplit>,
    pub execute_tail: f64,
    /// Under [`PROGRAM_FAST_ENV`] (the default), level 1's validator beside
    /// its execution: its seconds, and when the execution's end found it done
    /// (the driver's clock). `None` when off.
    pub validated_beside: Option<(f64, f64)>,
    /// Level 1's program: its walk's, its compile's and (off the knob) its
    /// validation's seconds before its execution could start.
    pub emit_split: (f64, f64, f64),
    /// Level 1's plan by part ([`super::whir_block::take_plan_split`]): frame,
    /// local lookups, prepared tables, costs, receivers' check, probes.
    pub plan_split: [f64; 6],
    /// Level 1's parallel compile by part
    /// ([`super::compiler::take_compile_split`]): multiplicities, rows, join.
    pub compile_split: [f64; 3],
    /// When the builder's thread ended, its prepare and packs done (the
    /// driver's clock); `None` when level 1 was prepared after the base.
    pub builder_end: Option<f64>,
    /// Level 1's lean round tables filled beside its execution
    /// (`LAMBDA_VM_WLFM_LEAN_BESIDE`): what the fill did, and the seconds
    /// level 1's thread waited for it once the execution had ended. `None`
    /// when off.
    pub lean_beside: Option<(super::b3l_tables::LeanBesideStats, f64)>,
    /// Whether level 1's fill took that prefill, and its split: the chips'
    /// seconds, the lean part's; and the lean part's own: the lean BLAKE3
    /// tables', the one-output selects' and cap tables'.
    pub lean_prefilled: bool,
    pub fill_split: (f64, f64),
    pub lean_parts: (f64, f64),
    /// Under [`STREAM_NEXT_ENV`], level 2's execution streamed over level 1's
    /// prove; `None` when it ran after it.
    pub next_stream: Option<NextStream>,
    /// When level 2's stacks' commit started, early or not
    /// ([`COMMIT_NEXT_EARLY_ENV`]); `None` without a level 2.
    pub next_commit: Option<NextCommit>,
    /// Under [`FILL_BESIDE_ENV`], level 1's fill on its own thread beside the
    /// stacks' commit: its seconds and when it ended (the driver's clock).
    /// `None` when the driver filled after the commit.
    pub fill_beside: Option<(f64, f64)>,
    /// The card's reading at the base's end ([`settle_base_end`]): its
    /// seconds, and whether it trimmed the pool.
    pub settle_base_end: Option<(f64, bool)>,
    pub levels: Vec<LevelClock>,
    /// Each level's prove stamps ([`super::wlfm::take_wlfm_stamps`]): its
    /// host preparation, phases and groups.
    pub stamps: Vec<super::wlfm::WlfmStamps>,
    /// Each level's build: where its stacks went, its host part's and its
    /// commits' seconds, its prepared cells, and when its commits ended (the
    /// driver's clock).
    pub builds: Vec<LevelBuild>,
    /// Each level's instructions and lean compressions (its program is let
    /// go once the level is proved).
    pub sizes: Vec<(usize, usize)>,
    /// Whether the process's peak resident set was started again before the
    /// base ([`reset_peak_rss`]).
    pub peak_reset: bool,
    /// The allocator purges: at the base's end, on a thread beside level 1;
    /// after the last level, before the driver returns; after level 1's fill,
    /// on a thread beside its prove; at the driver's start, before its clock.
    /// Each its seconds and jemalloc's resident bytes before and after.
    pub purges: [Option<crate::alloc_purge::Purge>; 4],
    /// The process's resident set at named points (bytes; Linux).
    pub resident: Vec<(&'static str, u64)>,
    /// The card at named points, each after a drain and a trim
    /// ([`card_settle`]).
    pub card: Vec<(String, CardReading)>,
    /// The levels' card pad held ([`CARD_PAD_ENV`]), in bytes.
    pub card_pad: u64,
}

/// One level's build, as the driver ran it.
#[derive(Clone, Copy, Debug)]
pub struct LevelBuild {
    pub place: StackPlace,
    pub prepare: f64,
    /// The prepare's seconds by part: `[groups, columns, stacks]`.
    pub prepare_split: [f64; 3],
    pub commit: f64,
    pub prefix: f64,
    pub cells: u64,
    pub committed: f64,
}

/// A block proved to its last level.
pub struct LevelsRun {
    /// The base proof, of either kind (`None` local argues in today's format).
    pub base_proof: crate::block_whir::AnyBlockWhirProof,
    pub base_stamps: crate::block_whir::BlockStamps,
    pub plan: WhirBlockPlan,
    pub levels: Vec<Level>,
    pub proofs: Vec<WlfmProof>,
    pub times: LevelsTimes,
}

/// The arena pieces level 1's streamed execution lands: the front, then the
/// groups in group order, at most 63 pieces of consecutive groups (the
/// streamed execution's 64). Returns each piece's arena range and its groups.
pub(crate) fn level1_pieces(marks: &[u32]) -> Vec<(std::ops::Range<u32>, std::ops::Range<usize>)> {
    let groups = marks.len() - 1;
    let per = groups.div_ceil(63).max(1);
    let mut out = vec![(0..marks[0], 0..0)];
    let mut g = 0usize;
    while g < groups {
        let hi = (g + per).min(groups);
        out.push((marks[g]..marks[hi], g..hi));
        g = hi;
    }
    out
}

/// The compressions level 1's waves wrote since the last call, handed to the
/// lean fill beside the execution (`to`, when it runs).
fn hand_on_lean<H: super::hash::LfmHasher + Sync>(
    ex: &mut super::executor::StreamedExecution<'_, '_, H>,
    to: Option<&std::sync::mpsc::Sender<Vec<(u32, super::b3l::B3lValues)>>>,
) {
    if let Some(to) = to {
        let wave = ex.take_lean_tap();
        if !wave.is_empty() {
            // A closed receiver is the fill's thread gone: its join says why.
            let _ = to.send(wave);
        }
    }
}

/// ★ A block proved to the end of its level chain: the base proof
/// (`base_options`, `format`, `options`), the plan derived the moment its
/// statement is final (one leaf over every group), level 1's program emitted
/// and EXECUTED BESIDE PHASE B (its arena's pieces landed as phase B finishes
/// each group, [`StreamedExecution::new_pieced`]), then level 1 filled and
/// proved, and each next level executed over the one below and proved. The
/// proofs are the ones [`prove_levels`] makes over the finished base; only
/// when the work runs moves. `beside = false` is the control: level 1's
/// groups land only once the base has ended.
///
/// A level's prepared stacks are its program's instruction columns,
/// gigabytes at a level's size. They are committed on the CARD, never on the
/// host while there is one ([`stack_place`]): held through the prove where
/// they fit beside a group's commit, else committed and let go at once and
/// revived at each group's opening. Level 1's are committed the moment the
/// base ends, while level 1's execution finishes on the host; each later
/// level's program and columns are prepared on the host while the level
/// below proves, and its stacks committed while its own execution runs.
/// `build_beside` prepares level 1's columns on a thread of its own the
/// moment its program is emitted, beside phase B; `false` prepares them
/// after the base. The same roots in every place, so the same proofs.
///
/// Memory: the process's peak resident set is started again before the base
/// ([`reset_peak_rss`]), the allocator's freed pages are handed back at the
/// base's end (beside level 1) and after the last level, each level's
/// program is let go once it is filled and its prover-side stacks once it is
/// proved. The card is drained and its pool trimmed ([`card_settle`]) before
/// and after level 1's stacks are committed and after each level's prove, so
/// a level's group commits find the card's memory, not the pool's leftovers.
///
/// Under [`WlfmFamily::B3`] the base is the 7-round BLAKE3 base
/// (`format.zf.base` must be `BaseFormat::B3_WHIR`, [`check_base`]), admitted
/// for this prove alone because its levels are BLAKE3 too
/// ([`crate::base_only::with_blake3_levels`]).
///
/// [`StreamedExecution::new_pieced`]: super::executor::StreamedExecution::new_pieced
#[allow(clippy::too_many_arguments)]
pub fn prove_block_levels(
    elf: &[u8],
    input: &[u8],
    base_options: &crate::ProofOptions,
    format: &crate::block_whir::BlockFormat,
    options: &crate::block_whir::BlockOptions,
    family: WlfmFamily,
    stop: LevelStop,
    beside: bool,
    build_beside: bool,
) -> Result<LevelsRun, String> {
    use super::executor::{ArenaPiece, StreamedExecution};
    use super::word::LfmWord;

    crate::base_only::refuse_recursion()?;
    check_base(&format.zf.base, family)?;
    // The last level's own format, read once (the process's knobs over the
    // family's pinned format: what the public verifier names unless a knob
    // turns a lever off), and refused now, before the base proves, when the
    // family cannot take it.
    let last = super::last_level::last_level(family)?;
    check_last_level_family(last, family)?;
    // Level 1's groups arrive as a streamed base's phase B opens them; the
    // whole-run base prover opens them without telling anyone.
    if options.window_log2.is_none() {
        return Err(
            "the levels driver lands level 1's groups as a streamed base opens them: the block \
             options must stream (window_log2)"
                .into(),
        );
    }
    let pool = program_pool()?;
    let pool = pool.as_ref();
    // What the process freed before this block (an earlier block, its
    // verification) goes back to the OS first, so the peak this block's base
    // reads starts from what is live, not from pages the allocator keeps.
    let keep = keep_memory();
    if keep {
        eprintln!(
            "LEVELS KEEP MEMORY: the purge and the card trim before the clock skipped (a test probe)"
        );
    }
    let purge_at_start = if keep {
        None
    } else {
        crate::alloc_purge::purge_now()
    };
    card_pad_bytes()?;
    card_budget_pct()?;
    let _pad = PadRelease;
    let _budget = BudgetRelease;
    let peak_reset = reset_peak_rss();
    // What an earlier block left in the card's pool goes back too.
    let at_start = if keep { None } else { card_settle("start") };
    let t0 = std::time::Instant::now();
    let now = || t0.elapsed().as_secs_f64();
    let (stx, srx) = mpsc::channel::<Stated>();
    let stx = std::sync::Mutex::new(Some(stx));
    let plan_cell: std::sync::OnceLock<WhirBlockPlan> = std::sync::OnceLock::new();
    let observe = |statement: crate::block_whir::BlockStatement<'_>,
                   roots: &[crate::block_whir::PreparedRoots]| {
        if let Some(tx) = stx.lock().unwrap_or_else(|e| e.into_inner()).as_ref() {
            let _ = tx.send(Stated::Final((statement.to_owned(), roots.to_vec())));
        }
    };
    // The base's predicted statement (`LAMBDA_VM_LEVELS_STATEMENT_EARLY`),
    // stamped with the driver's clock.
    let observe_early = |statement: crate::block_whir::BlockStatement<'_>,
                         roots: &[crate::block_whir::PreparedRoots]| {
        if let Some(tx) = stx.lock().unwrap_or_else(|e| e.into_inner()).as_ref() {
            let _ = tx.send(Stated::Early((statement.to_owned(), roots.to_vec()), now()));
        }
    };
    let early_note: std::sync::OnceLock<EarlyNote> = std::sync::OnceLock::new();
    let early_note_ref = &early_note;
    // A group's words need the plan; a group that finishes before it is
    // derived waits in the channel as its parts, here as its words once it is.
    type Parts = (
        usize,
        Vec<multilinear::whir_commit::Commitment>,
        Vec<crate::tables::types::FEE>,
        Vec<stark::multilinear_table::TableProof<crate::tables::types::GoldilocksExtension>>,
        Option<stark::multilinear_table::BatchedArgue<crate::tables::types::GoldilocksExtension>>,
        Option<
            stark::multilinear_table::batched::LocalArgue<
                crate::tables::types::GoldilocksExtension,
            >,
        >,
        multilinear::stacked_eval::StackedProof<
            crate::tables::types::GoldilocksField,
            crate::tables::types::GoldilocksExtension,
        >,
        Option<
            multilinear::stacked_eval::StackedProof<
                crate::tables::types::GoldilocksField,
                crate::tables::types::GoldilocksExtension,
            >,
        >,
    );
    let (ptx, prx) = mpsc::channel::<Parts>();
    let ptx = std::sync::Mutex::new(Some(ptx));
    let on_group = |opened: stark::multilinear_block::GroupOpened<'_, _, _>| {
        if let Some(tx) = ptx.lock().unwrap_or_else(|e| e.into_inner()).as_ref() {
            let _ = tx.send((
                opened.group,
                opened.roots.to_vec(),
                opened.commit_ood.to_vec(),
                opened.tables.to_vec(),
                opened.argue.cloned(),
                opened.local.cloned(),
                opened.opening.clone(),
                opened.prepared.cloned(),
            ));
        }
    };
    let (elf_ref, opts_ref, format_ref) = (elf, base_options, format);
    let plan_ref = &plan_cell;
    let mut times = LevelsTimes::default();
    // What level 1's thread hands back: when the plan and the program were
    // ready, its execution (with the lean fill beside it), or the execution
    // filled ([`Level1Out`]), when it could start and when it ended, and its
    // waves before the base ended.
    type Level1 = (
        f64,
        f64,
        Level1Out,
        f64,
        f64,
        usize,
        usize,
        f64,
        Option<(f64, f64)>,
        (f64, f64, f64),
        Level1Landing,
        (Option<[f64; 6]>, Option<[f64; 3]>),
    );
    // Read on the driver's thread (a test's override is that thread's).
    let fill_beside_on = fill_beside();
    let fill_parts_on = fill_parts();
    let commit_next_early_on = commit_next_early();
    let plan_lean_on = plan_lean();
    // The early statement: only where level 1 runs beside the base and
    // validates its program beside its execution.
    let statement_early_on = statement_early() && beside && program_fast();
    let next_beside_on = next_beside();
    let waves_in_order_on = waves_in_order();
    // The validator joined after the fill: only where it runs beside the
    // execution and the fill is level 1's thread's.
    let validate_after_fill_on = validate_after_fill() && fill_beside_on;
    // The builder's columns and packs on the whole pool from the base's
    // return (`LAMBDA_VM_LEVELS_BUILDER_WHOLE_POOL`): the driver raises this
    // as the base returns; the builder reads it between tables.
    let base_returned = super::wlfm::builder_whole_pool()
        .then(|| std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false)));
    let base_returned_ref = &base_returned;
    let whole_pool_note: std::sync::OnceLock<Option<super::wlfm::WholePoolNote>> =
        std::sync::OnceLock::new();
    let whole_pool_note_ref = &whole_pool_note;
    // Under the validator after the fill: the seconds its join waited.
    let after_fill_wait: std::sync::OnceLock<f64> = std::sync::OnceLock::new();
    let after_fill_wait_ref = &after_fill_wait;
    let stream_opts = StreamOpts {
        fill_parts: fill_parts_on,
        in_order: waves_in_order_on,
    };
    let late = settle_late() || super::wlfm_knobs::levels_untrimmed();
    // Level 1's lean round tables filled beside its execution: under the
    // knob, and only where they are held as their records.
    let lean_beside = super::wlfm::lean_beside()
        && super::b3l_tables::fill_hold() == super::b3l_tables::B3lHold::Records;
    let base_end: std::sync::OnceLock<f64> = std::sync::OnceLock::new();
    let base_end_ref = &base_end;
    // When the builder's thread ended (the driver's clock), under `build_beside`.
    let builder_end: std::sync::OnceLock<f64> = std::sync::OnceLock::new();
    let builder_end_ref = &builder_end;
    // Level 1's program, emitted by its thread and read by the builder's.
    let program_cell: std::sync::OnceLock<LfmProgram> = std::sync::OnceLock::new();
    let program_ref = &program_cell;
    let (btx, brx) = mpsc::channel::<()>();
    let result = std::thread::scope(|scope| {
        // The builder's thread (`build_beside`): level 1's stacks' columns,
        // on the host, as soon as its program exists. The card is the base's
        // until it ends: nothing is committed here.
        let builder = build_beside.then(|| {
            scope.spawn(move || -> Result<(WlfmPrebuild, f64), String> {
                brx.recv()
                    .map_err(|_| "level 1's program was never emitted")?;
                let program = program_ref.get().ok_or("the program is set")?;
                let plan = plan_ref.get().ok_or("the plan is set")?;
                let rounds = level_rounds(plan);
                // The whole-pool watch's note of the prebuild kept.
                let note = std::cell::Cell::new(None);
                let pre = prepare_as_level(stop, 1, last, family, |levers| {
                    let (pre, n) = prepare_level_watching(
                        program,
                        levers,
                        rounds,
                        base_returned_ref.clone(),
                        pool,
                    )?;
                    note.set(n);
                    Ok(pre)
                })?;
                let _ = whole_pool_note_ref.set(note.into_inner());
                Ok((pre, now()))
            })
        });
        // Level 1's thread: the plan, the program, then its execute, a piece
        // landed as soon as phase B has finished its groups.
        let level1 = scope.spawn(move || -> Result<Level1, String> {
            let beside_check = program_fast();
            // The plan and the program from a statement and its roots.
            let make = |statement: &crate::block_whir::OwnedBlockStatement,
                        roots: &[crate::block_whir::PreparedRoots]| {
                on_program_pool(pool, || -> Result<Made, String> {
                    let plan = level_plan_measured(
                        elf_ref,
                        opts_ref,
                        format_ref,
                        statement.view(),
                        family,
                        Some(roots),
                        !plan_lean_on,
                    )?;
                    let planned = now();
                    let plan_split = super::whir_block::take_plan_split();
                    let (program, marks, (walk, compiled)) =
                        plan.closed_block_program_marked_timed(beside_check)?;
                    let compile_split = super::compiler::take_compile_split();
                    Ok(Made {
                        plan,
                        planned,
                        plan_split,
                        program,
                        marks,
                        walk,
                        compiled,
                        compile_split,
                    })
                })
            };
            let first = srx.recv().map_err(|_| "the prover never stated")?;
            let made = match first {
                Stated::Final((statement, roots)) => make(&statement, &roots)?,
                // ★ The early statement (`LAMBDA_VM_LEVELS_STATEMENT_EARLY`):
                // the plan and the program made from it while phase A ends,
                // used only if the final statement is the same in every field
                // with the same roots; else made again from the final one, as
                // without the knob. Nothing is published before.
                Stated::Early((early, early_roots), stated_at) => {
                    let made_early = make(&early, &early_roots);
                    let made_at = now();
                    let t = std::time::Instant::now();
                    let Stated::Final((statement, roots)) =
                        srx.recv().map_err(|_| "the prover never stated")?
                    else {
                        return Err("the prover stated early twice".into());
                    };
                    let waited = t.elapsed().as_secs_f64();
                    let confirmed_at = now();
                    let difference = crate::block_whir::early_differs(
                        early.view(),
                        &early_roots,
                        statement.view(),
                        &roots,
                    );
                    let (made, why) = match (difference, made_early) {
                        (None, Ok(made)) => (made, None),
                        (None, Err(e)) | (Some(e), _) => (make(&statement, &roots)?, Some(e)),
                    };
                    if let Some(why) = &why {
                        EARLY_MISMATCHES.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                        eprintln!(
                            "LEVELS EARLY STATEMENT: not used ({why}); made from the final one"
                        );
                    }
                    let _ = early_note_ref.set(EarlyNote {
                        stated_at,
                        made_at,
                        confirmed_at,
                        waited,
                        used: why.is_none(),
                    });
                    made
                }
            };
            let Made {
                plan,
                planned,
                plan_split,
                program,
                marks,
                walk,
                compiled,
                compile_split,
            } = made;
            #[cfg(test)]
            let program = level1_program_fault_for_test(program);
            // Off the knob, validated before the execution, as always.
            let t = std::time::Instant::now();
            if !beside_check {
                validate_level(&program)
                    .map_err(|e| format!("level 1's program is not admissible: {e:?}"))?;
            }
            let emit_split = (walk, compiled, t.elapsed().as_secs_f64());
            let emitted = now();
            let _ = plan_ref.set(plan);
            let plan = plan_ref.get().ok_or("the plan is set")?;
            let _ = program_ref.set(program);
            let program = program_ref.get().ok_or("the program is set")?;
            let _ = btx.send(());
            // Under the knob, the admission validator runs beside the
            // execution; level 1 is refused at the execution's end if it fails.
            let checker = beside_check.then(|| spawn_validation(scope, program));
            let pieces = level1_pieces(&marks);
            let npieces = pieces.len();
            let hasher = program.hasher(<BlockFamily as ProofFamily>::SOCKET);
            let arena_pieces: Vec<ArenaPiece> = pieces
                .iter()
                .enumerate()
                .map(|(i, (range, _))| ArenaPiece {
                    arena: 0,
                    start: range.start,
                    len: range.end - range.start,
                    group: i,
                })
                .collect();
            let held: Vec<std::cell::OnceCell<Vec<LfmWord>>> =
                (0..npieces).map(|_| std::cell::OnceCell::new()).collect();
            let mut by_group: Vec<Option<Vec<LfmWord>>> = vec![None; marks.len() - 1];
            let placing = super::placement::placement();
            let usage_from = placing.then(|| {
                (
                    super::placement::thread_usage(),
                    super::placement::process_usage(),
                )
            });
            if placing {
                let _ = super::placement::take_jobs();
                super::placement::count_jobs(true);
            }
            let mut placed = super::placement::Placement::default();
            let execute_from = now();
            // The driver and lean fill keep their channel waits off the
            // workers. Only active work is installed in the pool.
            let mut ex = on_program_pool(pool, || {
                StreamedExecution::new_pieced_in_order(
                    program,
                    arena_pieces,
                    &hasher,
                    lean_beside,
                    bucket_waves(),
                    setup_fast(),
                    waves_in_order_on,
                )
            })
            .map_err(|e| format!("level 1's streamed execution: {e:?}"))?;
            let mut landing = Level1Landing {
                setup: now() - execute_from,
                ..Level1Landing::default()
            };
            // Under the knob, each wave's compressions go to a thread of their
            // own that makes the round records and counts as they come.
            let (lean_tx, lean_fill) = if lean_beside {
                let (tx, rx) = mpsc::channel::<Vec<(u32, super::b3l::B3lValues)>>();
                let fill = scope.spawn(move || {
                    let mut fill = super::b3l_tables::LeanFillBeside::new(program);
                    for wave in rx {
                        on_program_pool(pool, || fill.feed(&wave));
                    }
                    let stats = fill.stats();
                    (fill.finish(), stats)
                });
                (Some(tx), Some(fill))
            } else {
                (None, None)
            };
            hand_on_lean(&mut ex, lean_tx.as_ref());
            let mut landed = vec![false; npieces];
            let (mut waves_before, mut wave_secs_before) = (0usize, 0.0f64);
            // Beside phase B, each group as it comes; otherwise (the control)
            // every group once the base has ended.
            let parts: Box<dyn Iterator<Item = Parts>> = if beside {
                Box::new(prx.iter())
            } else {
                Box::new(prx.iter().collect::<Vec<_>>().into_iter())
            };
            for (g, roots, answers, tables, argue, local, opening, prepared) in parts {
                landing.arrivals.push((g, now()));
                if !landed[0] {
                    let front = super::whir_block::leaf_arena_under(
                        &roots,
                        &answers,
                        Vec::new(),
                        plan.wrap_hash(),
                    )
                    .into_iter()
                    .next()
                    .ok_or("no front")?;
                    held[0].set(front).map_err(|_| "the front landed twice")?;
                    let at = now();
                    let words = held[0].get().ok_or("the front")?;
                    on_program_pool(pool, || ex.land_piece(0, words))
                        .map_err(|e| format!("level 1's front: {e:?}"))?;
                    landing.front = (at, now() - at);
                    hand_on_lean(&mut ex, lean_tx.as_ref());
                    landed[0] = true;
                }
                let t_words = std::time::Instant::now();
                let words = super::whir_block::group_arena_words(
                    plan,
                    g,
                    &tables,
                    argue.as_ref(),
                    local.as_ref(),
                    &opening,
                    prepared.as_ref(),
                )?;
                landing.words.0 += 1;
                landing.words.1 += t_words.elapsed().as_secs_f64();
                *by_group
                    .get_mut(g)
                    .ok_or("a group the plan does not hold")? = Some(words);
                for (i, (_, groups)) in pieces.iter().enumerate().skip(1) {
                    if landed[i] || !groups.clone().all(|h| by_group[h].is_some()) {
                        continue;
                    }
                    let words: Vec<LfmWord> = groups
                        .clone()
                        .flat_map(|h| by_group[h].take().unwrap_or_default())
                        .collect();
                    held[i].set(words).map_err(|_| "a piece landed twice")?;
                    let cpu = placing.then(super::placement::this_cpu).flatten();
                    if placing {
                        let _ = super::placement::take_jobs();
                    }
                    let t = std::time::Instant::now();
                    let words = held[i].get().ok_or("a piece")?;
                    on_program_pool(pool, || ex.land_piece(i, words))
                        .map_err(|e| format!("level 1's piece {i}: {e:?}"))?;
                    hand_on_lean(&mut ex, lean_tx.as_ref());
                    if base_end_ref.get().is_none() {
                        waves_before += 1;
                        wave_secs_before += t.elapsed().as_secs_f64();
                    }
                    let secs = t.elapsed().as_secs_f64();
                    if placing {
                        placed.waves.push((cpu, super::placement::take_jobs()));
                    }
                    landing.landed.push((i, now() - secs, secs));
                    landed[i] = true;
                }
            }
            if !landed.iter().all(|&l| l) {
                return Err("phase B ended before level 1 had every group".into());
            }
            let execution = ex
                .finish()
                .map_err(|e| format!("level 1's execution: {e:?}"))?;
            let executed = now();
            if let Some((thread_from, process_from)) = usage_from {
                super::placement::count_jobs(false);
                placed.thread = thread_from
                    .zip(super::placement::thread_usage())
                    .map(|(a, b)| super::placement::usage_delta(a, b));
                placed.process = process_from
                    .zip(super::placement::process_usage())
                    .map(|(a, b)| super::placement::usage_delta(a, b));
                landing.placement = Some(placed);
            }
            // Under `LAMBDA_VM_LEVELS_VALIDATE_AFTER_FILL` the verdict is read
            // after the fill below, still before level 1 is handed on.
            let (validated, checker) = match checker {
                Some(handle) if !validate_after_fill_on => {
                    (Some((joined_validation(handle)?, now())), None)
                }
                other => (None, other),
            };
            drop(lean_tx);
            let lean = match lean_fill {
                Some(fill) => {
                    let (prefilled, stats) = fill
                        .join()
                        .map_err(|_| "level 1's lean fill thread panicked")?;
                    Some((prefilled, stats, now()))
                }
                None => None,
            };
            // Under the knob, level 1 filled here, beside the stacks' commit
            // the driver makes at the base's end. Phase B has closed the
            // parts channel, so this final fill can use the global pool.
            let execution = if fill_beside_on {
                let split = execution.split;
                let (prefilled, lean) = match lean {
                    Some((prefilled, stats, done)) => (
                        Some(prefilled.map_err(|e| format!("level 1: {e}"))?),
                        Some((stats, done)),
                    ),
                    None => (None, None),
                };
                let t = std::time::Instant::now();
                let filled = super::proof::lfm_fill_executed_parts(
                    program,
                    execution,
                    hasher,
                    executed - execute_from,
                    &Vec::new(),
                    super::wlfm::chip_records(),
                    prefilled,
                    fill_parts_on,
                );
                let fill = t.elapsed().as_secs_f64();
                Level1Out::Filled {
                    lean_prefilled: filled.lean_prefilled(),
                    fill_split: super::proof::take_fill_split().unwrap_or_default(),
                    lean_parts: super::proof::take_lean_parts().unwrap_or_default(),
                    filled,
                    split,
                    lean,
                    fill,
                    done: now(),
                }
            } else {
                Level1Out::Executed(execution, lean)
            };
            // The verdict, when it waited for the fill: read here, before
            // level 1 leaves this thread, so an inadmissible program is
            // refused before anything of it is proved.
            let validated = match checker {
                Some(handle) => {
                    let t = std::time::Instant::now();
                    let secs = joined_validation(handle)?;
                    let _ = after_fill_wait_ref.set(t.elapsed().as_secs_f64());
                    Some((secs, now()))
                }
                None => validated,
            };
            Ok((
                planned,
                emitted,
                execution,
                execute_from,
                executed,
                npieces,
                waves_before,
                wave_secs_before,
                validated,
                emit_split,
                landing,
                (plan_split, compile_split),
            ))
        });
        // A format that argues local lookups proves a proof of its own kind
        // (it carries every group's local argue), held as either kind.
        let prove_base = || {
            let early: Option<crate::block_whir::StatementObserver<'_>> =
                statement_early_on.then_some(&observe_early);
            if format.local_lookups() {
                crate::block_whir::prove_block_whir_local_observed_early(
                    elf,
                    input,
                    base_options,
                    format,
                    options,
                    &observe,
                    early,
                    &on_group,
                    None,
                )
                .map(|(mut proof, stamps)| {
                    let local = std::mem::take(&mut proof.local);
                    (proof.with_local(Some(local)), stamps)
                })
            } else {
                crate::block_whir::prove_block_whir_observed_early(
                    elf,
                    input,
                    base_options,
                    format,
                    options,
                    &observe,
                    early,
                    &on_group,
                )
                .map(|(proof, stamps)| (proof.with_local(None), stamps))
            }
        };
        // A BLAKE3 base is admitted for its own levels' prove alone: the
        // recursion over it is BLAKE3 too.
        let proved = match family {
            WlfmFamily::P1 => prove_base(),
            WlfmFamily::B3 => crate::base_only::with_blake3_levels(prove_base),
        };
        let _ = base_end_ref.set(now());
        if let Some(flag) = base_returned_ref {
            flag.store(true, std::sync::atomic::Ordering::Release);
        }
        let at_base = resident_now();
        *stx.lock().unwrap_or_else(|e| e.into_inner()) = None;
        *ptx.lock().unwrap_or_else(|e| e.into_inner()) = None;
        // The base's freed pages back to the OS, on a thread of their own
        // beside level 1: the levels' threads would not reuse them.
        let purge = proved
            .is_ok()
            .then(|| scope.spawn(crate::alloc_purge::purge_now));
        // The card is free: level 1's stacks are committed on it now, while
        // level 1's execution finishes on the host.
        let first = match builder {
            Some(b) => {
                let pre = b
                    .join()
                    .map_err(|_| "the builder's thread panicked".to_string());
                Some(pre.and_then(|pre| {
                    let (pre, builder_end) = pre?;
                    let _ = builder_end_ref.set(builder_end);
                    let prepare = (pre.secs(), pre.prepare_split());
                    let cells = pre.prepared_cells();
                    if proved.is_err() {
                        return Err("the base did not prove".to_string());
                    }
                    let (settled, settle_secs, trimmed) = settle_base_end(late);
                    take_levels_budget(settled.map(|c| c.total))?;
                    take_levels_pad()?;
                    let build = commit_level(pre, 1)?;
                    let committed = card_settle_levels("level 1 committed");
                    Ok((
                        build,
                        prepare,
                        cells,
                        now(),
                        [settled, committed],
                        (settle_secs, trimmed),
                    ))
                }))
            }
            None => None,
        };
        let level1 = level1.join();
        let purge = purge.and_then(|p| p.join().ok().flatten());
        (proved, level1, first, purge, at_base)
    });
    let (proved, level1, built_beside, purge, at_base) = result;
    times.peak_reset = peak_reset;
    times.purges[3] = purge_at_start;
    if let Some(card) = at_start {
        times.card.push(("start".to_string(), card));
    }
    times.purges[0] = purge;
    if let Some(r) = at_base {
        times.resident.push(("base end", r));
    }
    let (base_proof, base_stamps) = proved.map_err(|e| format!("the block proves: {e}"))?;
    times.base = base_end.get().copied().unwrap_or_else(now);
    times.builder_end = builder_end.get().copied();
    times.builder_whole_pool = whole_pool_note.get().copied().flatten();
    times.validated_after_fill = after_fill_wait.get().copied();
    times.early = early_note.get().copied();
    let (
        planned,
        emitted,
        execution,
        execute_from,
        executed,
        npieces,
        waves,
        wave_secs,
        validated,
        emit_split,
        landing,
        (plan_split, compile_split),
    ) = level1.map_err(|_| "level 1's thread panicked")??;
    times.landing = landing;
    times.plan_split = plan_split.unwrap_or_default();
    times.compile_split = compile_split.unwrap_or_default();
    times.validated_beside = validated;
    times.emit_split = emit_split;
    // A lean fill beside that was fed a compression twice, or missed one, is
    // refused here (or on level 1's thread, filled beside): never a wrong
    // count handed to the prove.
    let execution = match execution {
        Level1Out::Executed(execution, lean) => {
            times.level1_exec = Some(execution.split);
            let lean = match lean {
                Some((prefilled, stats, done)) => {
                    let prefilled = prefilled.map_err(|e| format!("level 1: {e}"))?;
                    times.lean_beside = Some((stats, (done - executed).max(0.0)));
                    Some(prefilled)
                }
                None => None,
            };
            Level1Ready::Executed(execution, lean)
        }
        Level1Out::Filled {
            filled,
            split,
            lean,
            lean_prefilled,
            fill_split,
            lean_parts,
            fill,
            done,
        } => {
            times.level1_exec = Some(split);
            if let Some((stats, lean_done)) = lean {
                times.lean_beside = Some((stats, (lean_done - executed).max(0.0)));
            }
            times.lean_prefilled = lean_prefilled;
            times.fill_split = fill_split;
            times.lean_parts = lean_parts;
            times.fill_beside = Some((fill, done));
            Level1Ready::Filled(filled, fill)
        }
    };
    let program = program_cell.into_inner().ok_or("no level 1 program")?;
    times.planned = planned;
    times.emitted = emitted;
    times.pieces = npieces;
    times.waves_before_base = waves;
    times.wave_secs_before_base = wave_secs;
    times.execute_tail = (executed - times.base).max(0.0);
    let plan = plan_cell.into_inner().ok_or("no plan")?;

    // Level 1's build: its stacks committed on the card at the base's end
    // (prepared beside phase B under `build_beside`), or prepared and
    // committed now.
    let (first, prepare, cells, committed, cards, settle) = match built_beside {
        Some(built) => built?,
        None => {
            let rounds = level_rounds(&plan);
            let pre = prepare_as_level(stop, 1, last, family, |levers| {
                prepare_level_as(&program, levers, rounds)
            })?;
            let (prepare, cells) = ((pre.secs(), pre.prepare_split()), pre.prepared_cells());
            let (settled, settle_secs, trimmed) =
                settle_base_end(settle_late() || super::wlfm_knobs::levels_untrimmed());
            take_levels_budget(settled.map(|c| c.total))?;
            take_levels_pad()?;
            let build = commit_level(pre, 1)?;
            let committed = card_settle_levels("level 1 committed");
            (
                build,
                prepare,
                cells,
                now(),
                [settled, committed],
                (settle_secs, trimmed),
            )
        }
    };
    times.settle_base_end = Some(settle);
    times.card_pad = levels_pad_held();
    for (at, card) in ["base end", "level 1 committed"].into_iter().zip(cards) {
        if let Some(card) = card {
            times.card.push((at.to_string(), card));
        }
    }
    times.built = committed;
    times.builds.push(LevelBuild {
        place: first.place(),
        prepare: prepare.0,
        prepare_split: prepare.1,
        commit: first.secs().1,
        prefix: first.chip_prefix_secs(),
        cells,
        committed,
    });
    if let Some(r) = resident_now() {
        times.resident.push(("level 1 built", r));
    }
    let mut levels = vec![Level {
        program,
        build: first,
    }];

    // Level 1: fill and prove; then each level over the one below, its
    // program and columns prepared while the one below proves, its stacks
    // committed on the card while its own execution runs.
    let mut proofs: Vec<WlfmProof> = Vec::with_capacity(2);
    // Read on the driver's thread (a test's override is that thread's).
    let stream_next_on = stream_next();
    std::thread::scope(|scope| -> Result<(), String> {
        let spawn_next = |build: &WlfmBuild, count: usize| {
            let child = child_of_build(&plan, build);
            level_more(stop, count, build)
                .then(|| scope.spawn(move || prepare_next(child, family, stop, count + 1, last)))
        };
        // Level `k + 1`'s stacks committed on a thread of their own: the build,
        // when the commit ended, and the card read after it.
        let spawn_commit = |pre: WlfmPrebuild, k: usize| {
            scope.spawn(move || {
                commit_level(pre, k + 1).map(|build| {
                    (
                        build,
                        now(),
                        card_settle_levels(&format!("level {} committed", k + 1)),
                    )
                })
            })
        };
        // Under the knob, level 2 executes beside level 1's prove, its arena
        // landing piece by piece as level 1 opens its groups.
        let mut streamed: Option<std::thread::ScopedJoinHandle<'_, Result<StreamedNext, String>>> =
            None;
        let (piece_tx, piece_rx) = mpsc::channel::<(usize, Vec<LfmWord>)>();
        let mut piece_rx = Some(piece_rx);
        // Under [`commit_next_early`], level 2's prebuild comes back on its
        // own as soon as it exists, to be committed when level 1's prove
        // returns rather than after level 2's fill.
        let mut early_rx: Option<mpsc::Receiver<WlfmPrebuild>> = None;
        let mut early_channel = || {
            commit_next_early_on.then(|| {
                let (tx, rx) = mpsc::sync_channel::<WlfmPrebuild>(1);
                early_rx = Some(rx);
                tx
            })
        };
        // Under [`next_beside`], level 2's thread starts here with its program
        // alone and prepares beside its execution; otherwise its program and
        // preparation run here and its stream waits for both.
        let beside = next_beside_on && stream_next_on && level_more(stop, 1, &levels[0].build);
        let mut next = if beside {
            let child = child_of_build(&plan, &levels[0].build);
            let early_tx = early_channel();
            let rx = piece_rx.take().ok_or("level 2's pieces")?;
            streamed = Some(scope.spawn(move || {
                stream_next_level_beside(
                    child,
                    family,
                    stop,
                    2,
                    last,
                    rx,
                    t0,
                    stream_opts,
                    early_tx,
                )
            }));
            None
        } else {
            spawn_next(&levels[0].build, 1)
        };
        let (filled, fill) = match execution {
            Level1Ready::Executed(execution, lean) => {
                let t = std::time::Instant::now();
                let filled = super::proof::lfm_fill_executed_parts(
                    &levels[0].program,
                    execution,
                    levels[0].build.artifacts.hasher,
                    executed - execute_from,
                    &Vec::new(),
                    super::wlfm::chip_records(),
                    lean,
                    fill_parts_on,
                );
                let fill = t.elapsed().as_secs_f64();
                times.lean_prefilled = filled.lean_prefilled();
                times.fill_split = super::proof::take_fill_split().unwrap_or_default();
                times.lean_parts = super::proof::take_lean_parts().unwrap_or_default();
                (filled, fill)
            }
            Level1Ready::Filled(filled, fill) => (filled, fill),
        };
        // Level 1's program let go: here, or under the knob on the purge's
        // thread below, beside the prove.
        let t_release = std::time::Instant::now();
        let free_beside_on = free_beside();
        let taken = take_program(&mut levels[0].program, &mut times.sizes);
        let taken = match free_beside_on {
            true => Some(taken),
            false => {
                drop(taken);
                None
            }
        };
        times.release = (t_release.elapsed().as_secs_f64(), free_beside_on);
        // What the execution and the program held, back to the OS on a
        // thread of its own beside the prove.
        let purge_filled = scope.spawn(move || {
            drop(taken);
            crate::alloc_purge::purge_now()
        });
        if let Some(r) = resident_now() {
            times.resident.push(("level 1 filled", r));
        }
        let _ = super::wlfm::take_wlfm_stamps();
        if !beside {
            if stream_next_on
                && let Some(prepared) = next.take()
                && let Some(rx) = piece_rx.take()
            {
                let early_tx = early_channel();
                streamed = Some(
                    scope.spawn(move || stream_next_level(prepared, rx, t0, stream_opts, early_tx)),
                );
            } else {
                drop(piece_rx.take());
            }
        }
        let streaming = streamed.is_some();
        let child = child_of(&plan, &levels[0]);
        let public = filled.public_words().to_vec();
        let front_sent = std::sync::atomic::AtomicBool::new(false);
        let observe_failed: std::sync::Mutex<Option<String>> = std::sync::Mutex::new(None);
        let observe = |opened: stark::multilinear_block::GroupOpened<'_, _, _>| {
            if !streaming {
                return;
            }
            let pieces = (|| -> Result<Vec<(usize, Vec<LfmWord>)>, String> {
                let mut out = Vec::with_capacity(2);
                if !front_sent.swap(true, std::sync::atomic::Ordering::Relaxed) {
                    let front = super::wlfm_leg::wlfm_leg_front(
                        &child,
                        &public,
                        opened.roots,
                        opened.commit_ood,
                    )?;
                    out.push((0, front));
                }
                let argue = opened.argue.ok_or("level 1 argues batched")?;
                let words = super::wlfm_leg::wlfm_leg_group(
                    &child,
                    opened.group,
                    argue,
                    opened.opening,
                    opened.prepared,
                )?;
                out.push((opened.group + 1, words));
                Ok(out)
            })();
            match pieces {
                Ok(pieces) => {
                    for piece in pieces {
                        let _ = piece_tx.send(piece);
                    }
                }
                Err(e) => {
                    let mut failed = observe_failed.lock().unwrap_or_else(|e| e.into_inner());
                    failed.get_or_insert(e);
                }
            }
        };
        let t = std::time::Instant::now();
        let proved = super::wlfm::prove_wlfm_traces_observed(
            &levels[0].build,
            filled,
            prefix_check(),
            WlfmProving::Streamed,
            &observe,
        )
        .map_err(|e| format!("level 1: {e}"));
        let level1_proved = now();
        drop(piece_tx);
        proofs.push(proved?);
        times.levels.push(LevelClock {
            execute_from,
            executed,
            fill,
            prove: t.elapsed().as_secs_f64(),
            end: level1_proved,
        });
        if let Some(e) = observe_failed
            .into_inner()
            .unwrap_or_else(|e| e.into_inner())
        {
            return Err(format!("level 2's arena from level 1's groups: {e}"));
        }
        // Level 1's card is let go here, then level 2's stacks committed at
        // once ([`commit_next_early`]), beside the end of its execution and
        // its fill; or both after its execution, as without the knob.
        let settled_early = early_rx.is_some();
        let mut early_commit = None;
        if let Some(rx) = early_rx.take() {
            levels[0].build.release_prover_half();
            if let Some(card) = card_settle_levels("level 1 proved") {
                times.card.push(("level 1 proved".to_string(), card));
            }
            // A closed channel is the streamed thread gone without its
            // prebuild: its join below says why.
            if let Ok(pre) = rx.recv() {
                let stats = (pre.secs(), pre.prepare_split(), pre.prepared_cells());
                times.next_commit = Some(NextCommit {
                    from: now(),
                    early: true,
                    joined: None,
                });
                early_commit = Some((spawn_commit(pre, 1), stats));
            }
        }
        let mut streamed_next = match streamed {
            Some(handle) => Some(
                handle
                    .join()
                    .map_err(|_| "level 2's streamed execution panicked".to_string())?
                    .map_err(|e| format!("level 2 streamed: {e}"))?,
            ),
            None => None,
        };
        if streaming {
            let joined = now();
            let next = times.next_commit.get_or_insert_with(NextCommit::default);
            next.joined = Some(joined);
        }
        times
            .stamps
            .push(super::wlfm::take_wlfm_stamps().unwrap_or_default());
        if !settled_early {
            levels[0].build.release_prover_half();
        }
        times.purges[2] = purge_filled.join().ok().flatten();
        if !settled_early && let Some(card) = card_settle_levels("level 1 proved") {
            times.card.push(("level 1 proved".to_string(), card));
        }
        if let Some(r) = resident_now() {
            times.resident.push(("level 1 proved", r));
        }
        let mut k = 1usize;
        loop {
            let (program, pre, done) = match (streamed_next.take(), next.take()) {
                (Some((program, pre, done)), _) => (program, pre, done),
                (None, Some(handle)) => {
                    let (program, pre, _marks) = handle
                        .join()
                        .map_err(|_| format!("level {}'s preparation panicked", k + 1))??;
                    (program, Some(pre), None)
                }
                (None, None) => break,
            };
            let execute_from = done.as_ref().map_or_else(now, |(_, t)| t.from);
            // Its stacks on the card while its execution runs on the host:
            // committed already ([`commit_next_early`]), or from here.
            let (committer, (prepare, prepare_split, cells)) = match (early_commit.take(), pre) {
                (Some(early), _) => early,
                (None, Some(pre)) => {
                    let stats = (pre.secs(), pre.prepare_split(), pre.prepared_cells());
                    if k == 1 {
                        let next = times.next_commit.get_or_insert_with(NextCommit::default);
                        next.from = now();
                    }
                    (spawn_commit(pre, k), stats)
                }
                (None, None) => {
                    return Err(format!(
                        "level {}'s prebuild was handed on and never committed",
                        k + 1
                    ));
                }
            };
            let (filled, executed, fill) = match done {
                Some((filled, stream)) => {
                    times.next_stream = Some(stream);
                    (filled, stream.executed, stream.fill)
                }
                None => {
                    let child = child_of(&plan, &levels[k - 1]);
                    let arena = wlfm_leg_arena(&child, &proofs[k - 1])?;
                    let hasher = program.hasher(<BlockFamily as ProofFamily>::SOCKET);
                    let filled =
                        super::wlfm::execute_and_fill_unmerged(&program, &[arena], hasher)?;
                    (filled, now(), 0.0)
                }
            };
            let mut program = program;
            release_program(&mut program, &mut times.sizes);
            let (build, committed, card) = committer
                .join()
                .map_err(|_| format!("level {}'s commit panicked", k + 1))??;
            if let Some(card) = card {
                times
                    .card
                    .push((format!("level {} committed", k + 1), card));
            }
            times.builds.push(LevelBuild {
                place: build.place(),
                prepare,
                prepare_split,
                commit: build.secs().1,
                prefix: build.chip_prefix_secs(),
                cells,
                committed,
            });
            next = spawn_next(&build, k + 1);
            let t = std::time::Instant::now();
            proofs.push(
                super::wlfm::prove_wlfm_traces_as(
                    &build,
                    filled,
                    prefix_check(),
                    WlfmProving::Streamed,
                )
                .map_err(|e| format!("level {}: {e}", k + 1))?,
            );
            times.levels.push(LevelClock {
                execute_from,
                executed,
                fill,
                prove: t.elapsed().as_secs_f64(),
                end: now(),
            });
            times
                .stamps
                .push(super::wlfm::take_wlfm_stamps().unwrap_or_default());
            levels.push(Level { program, build });
            levels[k].build.release_prover_half();
            if let Some(card) = card_settle_levels(&format!("level {} proved", k + 1)) {
                times.card.push((format!("level {} proved", k + 1), card));
            }
            k += 1;
        }
        Ok(())
    })?;
    if let Some(r) = resident_now() {
        times.resident.push(("levels proved", r));
    }
    // Each level's chains' round form, as built: the base's under the round
    // form, today's otherwise ([`level_rounds`]).
    eprintln!(
        "LEVELS ROUNDS: {:?}",
        levels
            .iter()
            .map(|l| l.build.artifacts.rounds)
            .collect::<Vec<_>>()
    );
    // What the levels freed, back to the OS before the next block's base.
    times.purges[1] = crate::alloc_purge::purge_now();
    Ok(LevelsRun {
        base_proof,
        base_stamps,
        plan,
        levels,
        proofs,
        times,
    })
}
