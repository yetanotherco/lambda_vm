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
//! the statement) and the level below's artifacts ([`derive_levels`]): a
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
/// host.
pub fn level_plan(
    elf: &[u8],
    proof_options: &crate::ProofOptions,
    format: &crate::block_whir::BlockFormat,
    statement: crate::block_whir::BlockStatement<'_>,
    family: WlfmFamily,
    prepared_roots: Option<&[(usize, Vec<multilinear::whir_commit::Commitment>)]>,
) -> Result<WhirBlockPlan, String> {
    crate::base_only::refuse_recursion()?;
    check_base(&format.zf.base, family)?;
    let plan = match family {
        WlfmFamily::P1 => WhirBlockPlan::derive_with(
            elf,
            proof_options,
            format,
            statement,
            Some(1),
            super::whir_block::BLOCK_FAN_IN,
            prepared_roots,
        )?,
        WlfmFamily::B3 => {
            WhirBlockPlan::derive_blake3(elf, proof_options, format, statement, 1, prepared_roots)?
        }
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
pub(crate) fn prepare_level_beside(
    program: &LfmProgram,
    family: WlfmFamily,
    rounds: Option<u8>,
) -> Result<WlfmPrebuild, String> {
    match beside_nice().then(beside_pool).flatten() {
        Some(pool) => pool.install(|| prepare_level(program, family, rounds)),
        None => prepare_level(program, family, rounds),
    }
}

/// `LAMBDA_VM_LEVELS_VALIDATE_LATE=1`: level 1's program is checked by the
/// admission validator from the base's end, beside level 1's prove, instead of
/// beside its execution (which runs beside the base's phase B, where the host's
/// work slows the base). The driver returns no proof until the check has
/// passed: an inadmissible program still refuses the block, only later. The
/// verifier validates every level's program it derives either way
/// ([`super::whir_block::finish_program`]). Under [`program_fast`] only (off
/// it, the program is validated before its execution, as always). Off by
/// default (and for any word but `1`); read once a process.
pub const VALIDATE_LATE_ENV: &str = "LAMBDA_VM_LEVELS_VALIDATE_LATE";

std::thread_local! {
    /// A test's [`validate_late`] on this thread.
    static VALIDATE_LATE_FOR_TEST: std::cell::Cell<Option<bool>> = const { std::cell::Cell::new(None) };
}

/// Tests only: [`validate_late`] forced on this thread (`None` restores the
/// knob). The driver reads it on its own thread.
#[doc(hidden)]
pub fn set_validate_late_for_test(on: Option<bool>) {
    VALIDATE_LATE_FOR_TEST.with(|c| c.set(on));
}

/// Whether [`VALIDATE_LATE_ENV`] is on (or this thread's test override).
pub fn validate_late() -> bool {
    if let Some(on) = VALIDATE_LATE_FOR_TEST.with(std::cell::Cell::get) {
        return on;
    }
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        super::wlfm_knobs::knob_word_on(std::env::var(VALIDATE_LATE_ENV).ok().as_deref())
    })
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
        super::validator::validate(program)
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
pub(crate) fn prepare_level(
    program: &LfmProgram,
    family: WlfmFamily,
    rounds: Option<u8>,
) -> Result<WlfmPrebuild, String> {
    let hasher = program.hasher(<BlockFamily as ProofFamily>::SOCKET);
    let mut pre = prepare_wlfm_rounds(program, hasher, LEVEL_RATE, level_levers(family), rounds)?;
    if super::wlfm::chip_records() {
        pre.hold_chip_prefixes()?;
    }
    Ok(pre)
}

/// A prepared level committed where [`stack_place`] puts it.
pub(crate) fn commit_level(pre: WlfmPrebuild, level: usize) -> Result<WlfmBuild, String> {
    let place = stack_place(pre.prepared_cells(), level, stacks_forced()?);
    pre.commit(place)
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
fn prepare_next(child: WlfmChild, family: WlfmFamily) -> Result<PreparedNext, String> {
    let (program, marks) = wlfm_leg_program_marked(&child)?;
    let pre = prepare_level(&program, family, child.artifacts.rounds)?;
    Ok((program, pre, marks))
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
/// prove when it streamed ([`stream_next`]).
type StreamedNext = (
    LfmProgram,
    WlfmPrebuild,
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
    Ok(got)
}

/// Gives the levels' pad back.
fn release_levels_pad() {
    *LEVELS_PAD.lock().unwrap_or_else(|e| e.into_inner()) = None;
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

/// Puts the levels' budget in force ([`CARD_BUDGET_ENV`], 85 % by default) on
/// a card of `total` bytes: the budget now (unchanged without a card, or when
/// the process set its own card budget and the knob says nothing).
fn take_levels_budget(total: Option<u64>) -> Result<u64, String> {
    let (Some(pct), Some(total)) = (card_budget_pct()?, total) else {
        return Ok(multilinear::gpu::reserve_budget());
    };
    let bytes = total / 100 * pct;
    let base = multilinear::gpu::set_reserve_budget(bytes);
    let mut held = BASE_BUDGET.lock().unwrap_or_else(|e| e.into_inner());
    if held.is_none() {
        *held = Some(base);
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
}

/// Puts the base's budget back when the driver returns, by any path.
struct BudgetRelease;

impl Drop for BudgetRelease {
    fn drop(&mut self) {
        release_levels_budget();
    }
}

/// A filled level's program let go: its instructions and instruction
/// columns (their counts kept in `sizes`). The prove reads the build and the
/// filled traces, never the program; the heights stay.
fn release_program(p: &mut LfmProgram, sizes: &mut Vec<(usize, usize)>) {
    sizes.push((p.instrs.len(), p.groups.b3l.real_rows));
    p.instrs = Vec::new();
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
        group.data = Vec::new();
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
    let (pool_used, pool_peak, free, total) = multilinear::gpu::settle_card()?;
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

/// One level built from its program: under `family`'s levers at
/// [`LEVEL_RATE`], its chains under `rounds` ([`level_rounds`]), its
/// prepared stacks committed on the host when `on_host`.
pub(crate) fn build_level(
    program: LfmProgram,
    family: WlfmFamily,
    rounds: Option<u8>,
    on_host: bool,
) -> Result<Level, String> {
    let hasher = program.hasher(<BlockFamily as ProofFamily>::SOCKET);
    let place = if on_host {
        StackPlace::Host
    } else {
        StackPlace::Card
    };
    let build = prepare_wlfm_rounds(&program, hasher, LEVEL_RATE, level_levers(family), rounds)?
        .commit(place)?;
    Ok(Level { program, build })
}

/// Level `k + 1`'s view of level `k`: its artifacts and the top's word count.
/// The most pieces a streamed execution lands (one bit a piece).
const MAX_STREAM_PIECES: usize = 64;

/// Level 2 beside level 1's prove ([`stream_next`]): its program and prebuild
/// from `prepared` (joined here), and its execution streamed over the pieces
/// `pieces` hands on ([`stream_level_over_pieces`]); `None` for the execution
/// when its arena has more pieces than a streamed execution lands (level 2
/// then executes after level 1's prove, as without the knob).
fn stream_next_level(
    prepared: std::thread::ScopedJoinHandle<'_, Result<PreparedNext, String>>,
    pieces: mpsc::Receiver<(usize, Vec<LfmWord>)>,
    t0: std::time::Instant,
) -> Result<StreamedNext, String> {
    let (program, pre, marks) = prepared
        .join()
        .map_err(|_| "level 2's preparation panicked".to_string())??;
    if marks.len() > MAX_STREAM_PIECES {
        return Ok((program, pre, None));
    }
    let hasher = program.hasher(<BlockFamily as ProofFamily>::SOCKET);
    let streamed = stream_level_over_pieces(&program, &marks, hasher, pieces, t0)?;
    Ok((program, pre, Some(streamed)))
}

/// `program`'s execution over one arena cut at `marks` (each piece's end:
/// the front is piece 0, a child's group `g` is piece `g + 1`), each piece
/// landed as `pieces` hands it on, in any order, and filled once the last has
/// landed: [`super::wlfm::execute_and_fill_unmerged`]'s traces over the
/// pieces joined. The channel closing before every piece came refuses.
pub(super) fn stream_level_over_pieces(
    program: &LfmProgram,
    marks: &[u32],
    hasher: super::HasherKind,
    pieces: mpsc::Receiver<(usize, Vec<LfmWord>)>,
    t0: std::time::Instant,
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
        let mut ex = StreamedExecution::new_pieced_setup(
            program,
            arena_pieces,
            &hasher,
            false,
            bucket_waves(),
            setup_fast(),
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
    let filled = super::proof::lfm_fill_executed_holding(
        program,
        execution,
        hasher,
        times.executed - from,
        &Vec::new(),
        super::wlfm::chip_records(),
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
    }
}

/// ★ Every level of `plan`'s chain, its programs emitted and built: ONE
/// derivation for the prover (`on_host = false`: prepared commits on the
/// card) and the verifier (`on_host = true`). `plan` must be one leaf over
/// every group.
pub fn derive_levels(
    plan: &WhirBlockPlan,
    family: WlfmFamily,
    on_host: bool,
) -> Result<Vec<Level>, String> {
    derive_levels_to(plan, family, LevelStop::DEFAULT, on_host)
}

/// [`derive_levels`] under the chain stop `stop`.
pub fn derive_levels_to(
    plan: &WhirBlockPlan,
    family: WlfmFamily,
    stop: LevelStop,
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
    let rounds = level_rounds(plan);
    let mut levels = vec![build_level(
        plan.closed_block_program()?,
        family,
        rounds,
        on_host,
    )?];
    while more(&levels)? {
        let child = child_of(plan, levels.last().expect("a level"));
        let rounds = child.artifacts.rounds;
        levels.push(build_level(
            wlfm_leg_program(&child)?,
            family,
            rounds,
            on_host,
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
pub fn prove_levels(
    plan: &WhirBlockPlan,
    levels: &[Level],
    base: &BlockWhirProof,
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

/// ★ The block's verifier over its last level: every level derived on the
/// host from `plan` ([`derive_levels`]), `last` verified against the last
/// level's artifacts, and its words claiming the block.
pub fn verify_levels(
    plan: &WhirBlockPlan,
    family: WlfmFamily,
    last: &WlfmProof,
) -> Result<(), String> {
    verify_levels_to(plan, family, LevelStop::DEFAULT, last)
}

/// [`verify_levels`] under the chain stop `stop` (the verifier's constant).
pub fn verify_levels_to(
    plan: &WhirBlockPlan,
    family: WlfmFamily,
    stop: LevelStop,
    last: &WlfmProof,
) -> Result<(), String> {
    let levels = derive_levels_to(plan, family, stop, true)?;
    let top = levels.last().expect("a level");
    verify_wlfm(&top.build.artifacts, last, &last.public_words)
        .map_err(|e| format!("the last level does not verify: {e}"))?;
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
    /// Under [`VALIDATE_LATE_ENV`], level 1's validator from the base's end:
    /// its seconds, and when level 1's proved prove found it done.
    pub validated_late: Option<(f64, f64)>,
    /// Level 1's program: its walk's, its compile's and (off the knob) its
    /// validation's seconds before its execution could start.
    pub emit_split: (f64, f64, f64),
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
    /// Under [`FILL_BESIDE_ENV`], level 1's fill on its own thread beside the
    /// stacks' commit: its seconds and when it ended (the driver's clock).
    /// `None` when the driver filled after the commit.
    pub fill_beside: Option<(f64, f64)>,
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
    pub base_proof: BlockWhirProof,
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
    // Level 1's groups arrive as a streamed base's phase B opens them; the
    // whole-run base prover opens them without telling anyone.
    if options.window_log2.is_none() {
        return Err(
            "the levels driver lands level 1's groups as a streamed base opens them: the block \
             options must stream (window_log2)"
                .into(),
        );
    }
    // What the process freed before this block (an earlier block, its
    // verification) goes back to the OS first, so the peak this block's base
    // reads starts from what is live, not from pages the allocator keeps.
    let purge_at_start = crate::alloc_purge::purge_now();
    card_pad_bytes()?;
    card_budget_pct()?;
    let _pad = PadRelease;
    let _budget = BudgetRelease;
    let peak_reset = reset_peak_rss();
    // What an earlier block left in the card's pool goes back too.
    let at_start = card_settle("start");
    let t0 = std::time::Instant::now();
    let now = || t0.elapsed().as_secs_f64();
    type Statement = (
        crate::block_whir::OwnedBlockStatement,
        Vec<crate::block_whir::PreparedRoots>,
    );
    let (stx, srx) = mpsc::channel::<Statement>();
    let stx = std::sync::Mutex::new(Some(stx));
    let plan_cell: std::sync::OnceLock<WhirBlockPlan> = std::sync::OnceLock::new();
    let observe = |statement: crate::block_whir::BlockStatement<'_>,
                   roots: &[crate::block_whir::PreparedRoots]| {
        if let Some(tx) = stx.lock().unwrap_or_else(|e| e.into_inner()).as_ref() {
            let _ = tx.send((statement.to_owned(), roots.to_vec()));
        }
    };
    // A group's words need the plan; a group that finishes before it is
    // derived waits in the channel as its parts, here as its words once it is.
    type Parts = (
        usize,
        Vec<multilinear::whir_commit::Commitment>,
        Vec<crate::tables::types::FEE>,
        Vec<stark::multilinear_table::TableProof<crate::tables::types::GoldilocksExtension>>,
        Option<stark::multilinear_table::BatchedArgue<crate::tables::types::GoldilocksExtension>>,
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
    );
    // Read on the driver's thread (a test's override is that thread's).
    let fill_beside_on = fill_beside();
    // Level 1's lean round tables filled beside its execution: under the
    // knob, and only where they are held as their records.
    let lean_beside = super::wlfm::lean_beside()
        && super::b3l_tables::fill_hold() == super::b3l_tables::B3lHold::Records;
    let base_end: std::sync::OnceLock<f64> = std::sync::OnceLock::new();
    let base_end_ref = &base_end;
    // Read on the driver's thread (a test's override is that thread's).
    let validate_late_on = validate_late() && program_fast();
    // Level 1's program, emitted by its thread and read by the builder's.
    let program_cell: std::sync::OnceLock<LfmProgram> = std::sync::OnceLock::new();
    let program_ref = &program_cell;
    let (btx, brx) = mpsc::channel::<()>();
    let result = std::thread::scope(|scope| {
        // The builder's thread (`build_beside`): level 1's stacks' columns,
        // on the host, as soon as its program exists. The card is the base's
        // until it ends: nothing is committed here.
        let builder = build_beside.then(|| {
            scope.spawn(move || -> Result<WlfmPrebuild, String> {
                brx.recv()
                    .map_err(|_| "level 1's program was never emitted")?;
                let program = program_ref.get().ok_or("the program is set")?;
                let plan = plan_ref.get().ok_or("the plan is set")?;
                prepare_level_beside(program, family, level_rounds(plan))
            })
        });
        // Level 1's thread: the plan, the program, then its execute, a piece
        // landed as soon as phase B has finished its groups.
        let level1 = scope.spawn(move || -> Result<Level1, String> {
            let (statement, roots) = srx.recv().map_err(|_| "the prover never stated")?;
            let plan = level_plan(
                elf_ref,
                opts_ref,
                format_ref,
                statement.view(),
                family,
                Some(&roots),
            )?;
            let planned = now();
            let beside_check = program_fast();
            let (program, marks, (walk, compiled)) =
                plan.closed_block_program_marked_timed(beside_check)?;
            // Off the knob, validated before the execution, as always.
            let t = std::time::Instant::now();
            if !beside_check {
                super::validator::validate(&program)
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
            let checker =
                (beside_check && !validate_late_on).then(|| spawn_validation(scope, program));
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
            let mut ex = StreamedExecution::new_pieced_setup(
                program,
                arena_pieces,
                &hasher,
                lean_beside,
                bucket_waves(),
                setup_fast(),
            )
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
                        fill.feed(&wave);
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
            for (g, roots, answers, tables, argue, opening, prepared) in parts {
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
                    ex.land_piece(0, held[0].get().ok_or("the front")?)
                        .map_err(|e| format!("level 1's front: {e:?}"))?;
                    landing.front = (at, now() - at);
                    hand_on_lean(&mut ex, lean_tx.as_ref());
                    landed[0] = true;
                }
                let words = super::whir_block::group_arena_words(
                    plan,
                    g,
                    &tables,
                    argue.as_ref(),
                    &opening,
                    prepared.as_ref(),
                )?;
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
                    ex.land_piece(i, held[i].get().ok_or("a piece")?)
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
            let validated = match checker {
                Some(handle) => Some((joined_validation(handle)?, now())),
                None => None,
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
            // the driver makes at the base's end.
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
                let filled = super::proof::lfm_fill_executed_prefilled(
                    program,
                    execution,
                    hasher,
                    executed - execute_from,
                    &Vec::new(),
                    super::wlfm::chip_records(),
                    prefilled,
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
            ))
        });
        let prove_base = || {
            crate::block_whir::prove_block_whir_observed_groups(
                elf,
                input,
                base_options,
                format,
                options,
                &observe,
                &on_group,
            )
        };
        // A BLAKE3 base is admitted for its own levels' prove alone: the
        // recursion over it is BLAKE3 too.
        let proved = match family {
            WlfmFamily::P1 => prove_base(),
            WlfmFamily::B3 => crate::base_only::with_blake3_levels(prove_base),
        };
        let _ = base_end_ref.set(now());
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
                    let pre = pre?;
                    let prepare = (pre.secs(), pre.prepare_split());
                    let cells = pre.prepared_cells();
                    if proved.is_err() {
                        return Err("the base did not prove".to_string());
                    }
                    let settled = card_settle("base end");
                    take_levels_budget(settled.map(|c| c.total))?;
                    take_levels_pad()?;
                    let build = commit_level(pre, 1)?;
                    let committed = card_settle("level 1 committed");
                    Ok((build, prepare, cells, now(), [settled, committed]))
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
    ) = level1.map_err(|_| "level 1's thread panicked")??;
    times.landing = landing;
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
    let (first, prepare, cells, committed, cards) = match built_beside {
        Some(built) => built?,
        None => {
            let pre = prepare_level(&program, family, level_rounds(&plan))?;
            let (prepare, cells) = ((pre.secs(), pre.prepare_split()), pre.prepared_cells());
            let settled = card_settle("base end");
            take_levels_budget(settled.map(|c| c.total))?;
            take_levels_pad()?;
            let build = commit_level(pre, 1)?;
            let committed = card_settle("level 1 committed");
            (build, prepare, cells, now(), [settled, committed])
        }
    };
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
            level_more(stop, count, build).then(|| scope.spawn(move || prepare_next(child, family)))
        };
        let mut next = spawn_next(&levels[0].build, 1);
        let (filled, fill) = match execution {
            Level1Ready::Executed(execution, lean) => {
                let t = std::time::Instant::now();
                let filled = super::proof::lfm_fill_executed_prefilled(
                    &levels[0].program,
                    execution,
                    levels[0].build.artifacts.hasher,
                    executed - execute_from,
                    &Vec::new(),
                    super::wlfm::chip_records(),
                    lean,
                );
                let fill = t.elapsed().as_secs_f64();
                times.lean_prefilled = filled.lean_prefilled();
                times.fill_split = super::proof::take_fill_split().unwrap_or_default();
                times.lean_parts = super::proof::take_lean_parts().unwrap_or_default();
                (filled, fill)
            }
            Level1Ready::Filled(filled, fill) => (filled, fill),
        };
        // Under the knob, the program goes to the admission validator, on a
        // thread of its own beside level 1's prove, which lets it go once
        // checked; otherwise it is let go here.
        let late_check = if validate_late_on {
            let p = &levels[0].program;
            times.sizes.push((p.instrs.len(), p.groups.b3l.real_rows));
            let program = std::mem::replace(
                &mut levels[0].program,
                super::compiler::compile(super::builder::LfmBuilder::new().finish()),
            );
            Some(scope.spawn(move || -> Result<f64, String> {
                let t = std::time::Instant::now();
                super::validator::validate(&program)
                    .map_err(|e| format!("level 1's program is not admissible: {e:?}"))?;
                let secs = t.elapsed().as_secs_f64();
                drop(program);
                Ok(secs)
            }))
        } else {
            release_program(&mut levels[0].program, &mut times.sizes);
            None
        };
        // What the execution and the program held, back to the OS on a
        // thread of its own beside the prove.
        let purge_filled = scope.spawn(crate::alloc_purge::purge_now);
        if let Some(r) = resident_now() {
            times.resident.push(("level 1 filled", r));
        }
        let _ = super::wlfm::take_wlfm_stamps();
        // Under the knob, level 2 executes beside level 1's prove, its arena
        // landing piece by piece as level 1 opens its groups.
        let mut streamed: Option<std::thread::ScopedJoinHandle<'_, Result<StreamedNext, String>>> =
            None;
        let (piece_tx, piece_rx) = mpsc::channel::<(usize, Vec<LfmWord>)>();
        if stream_next_on && let Some(prepared) = next.take() {
            streamed = Some(scope.spawn(move || stream_next_level(prepared, piece_rx, t0)));
        } else {
            drop(piece_rx);
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
        // No proof leaves before the program it proves has passed admission.
        if let Some(handle) = late_check {
            let secs = handle
                .join()
                .map_err(|_| "level 1's validator panicked".to_string())??;
            times.validated_late = Some((secs, now()));
        }
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
        let mut streamed_next = match streamed {
            Some(handle) => Some(
                handle
                    .join()
                    .map_err(|_| "level 2's streamed execution panicked".to_string())?
                    .map_err(|e| format!("level 2 streamed: {e}"))?,
            ),
            None => None,
        };
        times
            .stamps
            .push(super::wlfm::take_wlfm_stamps().unwrap_or_default());
        levels[0].build.release_prover_half();
        times.purges[2] = purge_filled.join().ok().flatten();
        if let Some(card) = card_settle("level 1 proved") {
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
                    (program, pre, None)
                }
                (None, None) => break,
            };
            let execute_from = done.as_ref().map_or_else(now, |(_, t)| t.from);
            let (prepare, prepare_split, cells) =
                (pre.secs(), pre.prepare_split(), pre.prepared_cells());
            // Its stacks on the card while its execution runs on the host.
            let committer = scope.spawn(move || {
                commit_level(pre, k + 1).map(|build| {
                    (
                        build,
                        now(),
                        card_settle(&format!("level {} committed", k + 1)),
                    )
                })
            });
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
            if let Some(card) = card_settle(&format!("level {} proved", k + 1)) {
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
