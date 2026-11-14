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
//! caller names (a verifier constant: [`WlfmFamily::P1`] today, whose in-guest
//! arm exists). The plan's own family must be the levels' — one hash end to
//! end.

use super::compiler::LfmProgram;
use super::whir_block::{WhirBlockPlan, block_leaf_arena, top_claims};
use super::wlfm::{
    WlfmBuild, WlfmFamily, WlfmLevers, WlfmProof, WlfmProving, WlfmRate, build_wlfm_under,
    prove_wlfm_as, verify_wlfm,
};
use super::wlfm_leg::{WlfmChild, wlfm_leg_arena, wlfm_leg_program};
use crate::block_whir::BlockWhirProof;
use crate::hash_pin::{BlockFamily, ProofFamily};

/// The rate every level proves at: the base's, blowup 2.
pub const LEVEL_RATE: WlfmRate = WlfmRate::Two;

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

/// The family whose in-guest arm the plan's programs replay: one hash end to
/// end, so the levels' must be it.
fn check_family(plan: &WhirBlockPlan, family: WlfmFamily) -> Result<(), String> {
    let _ = plan;
    if family != WlfmFamily::P1 {
        return Err(format!(
            "the levels run the plan's in-guest arm, Poseidon1's; {family:?} has none yet"
        ));
    }
    Ok(())
}

fn build_level(program: LfmProgram, family: WlfmFamily, on_host: bool) -> Result<Level, String> {
    let hasher = program.hasher(<BlockFamily as ProofFamily>::SOCKET);
    let build = build_wlfm_under(&program, hasher, LEVEL_RATE, level_levers(family), on_host)?;
    Ok(Level { program, build })
}

/// Level `k + 1`'s view of level `k`: its artifacts and the top's word count.
pub fn child_of(plan: &WhirBlockPlan, level: &Level) -> WlfmChild {
    WlfmChild {
        artifacts: level.build.artifacts.clone(),
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
    let mut levels = vec![build_level(plan.closed_block_program()?, family, on_host)?];
    while more(&levels)? {
        let child = child_of(plan, levels.last().expect("a level"));
        levels.push(build_level(wlfm_leg_program(&child)?, family, on_host)?);
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
        let proof = prove_wlfm_as(&level.program, &level.build, &arenas, WlfmProving::Streamed)
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
    pub execute_tail: f64,
    pub levels: Vec<LevelClock>,
}

/// A block proved to its last level.
pub struct LevelsRun {
    pub base_proof: BlockWhirProof,
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

/// ★ A block proved to the end of its level chain: the base proof
/// (`base_options`, `format`, `options`), the plan derived the moment its
/// statement is final (one leaf over every group), level 1's program emitted
/// and EXECUTED BESIDE PHASE B (its arena's pieces landed as phase B finishes
/// each group, [`StreamedExecution::new_pieced`]), then level 1 filled and
/// proved, and each next level executed over the one below and proved. The
/// proofs are the ones [`prove_levels`] makes over the finished base; only
/// when the work runs moves.
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
) -> Result<LevelsRun, String> {
    use super::executor::{ArenaPiece, StreamedExecution};
    use super::word::LfmWord;
    use std::sync::mpsc;

    crate::base_only::refuse_recursion()?;
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
    // ready, the program, its execution, when it could start and when it
    // ended, and its waves before the base ended.
    type Level1 = (
        f64,
        f64,
        LfmProgram,
        super::executor::LfmExecution,
        f64,
        f64,
        usize,
        usize,
        f64,
    );
    let base_end: std::sync::OnceLock<f64> = std::sync::OnceLock::new();
    let base_end_ref = &base_end;
    let result = std::thread::scope(|scope| {
        // Level 1's thread: the plan, the program, then its execute, a piece
        // landed as soon as phase B has finished its groups.
        let level1 = scope.spawn(move || -> Result<Level1, String> {
            let (statement, roots) = srx.recv().map_err(|_| "the prover never stated")?;
            let plan = WhirBlockPlan::derive_with(
                elf_ref,
                opts_ref,
                format_ref,
                statement.view(),
                Some(1),
                super::whir_block::BLOCK_FAN_IN,
                Some(&roots),
            )?;
            let planned = now();
            let (program, marks) = plan.closed_block_program_marked()?;
            let emitted = now();
            let _ = plan_ref.set(plan);
            let plan = plan_ref.get().ok_or("the plan is set")?;
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
            let execute_from = now();
            let mut ex = StreamedExecution::new_pieced(&program, arena_pieces, &hasher)
                .map_err(|e| format!("level 1's streamed execution: {e:?}"))?;
            let mut landed = vec![false; npieces];
            let (mut waves_before, mut wave_secs_before) = (0usize, 0.0f64);
            for (g, roots, answers, tables, argue, opening, prepared) in prx.iter() {
                if !landed[0] {
                    let front = super::whir_block::leaf_arena(&roots, &answers, Vec::new())
                        .into_iter()
                        .next()
                        .ok_or("no front")?;
                    held[0].set(front).map_err(|_| "the front landed twice")?;
                    ex.land_piece(0, held[0].get().ok_or("the front")?)
                        .map_err(|e| format!("level 1's front: {e:?}"))?;
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
                    let t = std::time::Instant::now();
                    ex.land_piece(i, held[i].get().ok_or("a piece")?)
                        .map_err(|e| format!("level 1's piece {i}: {e:?}"))?;
                    if base_end_ref.get().is_none() {
                        waves_before += 1;
                        wave_secs_before += t.elapsed().as_secs_f64();
                    }
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
            Ok((
                planned,
                emitted,
                program,
                execution,
                execute_from,
                executed,
                npieces,
                waves_before,
                wave_secs_before,
            ))
        });
        let proved = crate::block_whir::prove_block_whir_observed_groups(
            elf,
            input,
            base_options,
            format,
            options,
            &observe,
            &on_group,
        );
        let _ = base_end_ref.set(now());
        *stx.lock().unwrap_or_else(|e| e.into_inner()) = None;
        *ptx.lock().unwrap_or_else(|e| e.into_inner()) = None;
        (proved, level1.join())
    });
    let (proved, level1) = result;
    let (base_proof, _stamps) = proved.map_err(|e| format!("the block proves: {e}"))?;
    times.base = base_end.get().copied().unwrap_or_else(now);
    let (planned, emitted, program, execution, execute_from, executed, npieces, waves, wave_secs) =
        level1.map_err(|_| "level 1's thread panicked")??;
    times.planned = planned;
    times.emitted = emitted;
    times.pieces = npieces;
    times.waves_before_base = waves;
    times.wave_secs_before_base = wave_secs;
    times.execute_tail = (executed - times.base).max(0.0);
    let plan = plan_cell.into_inner().ok_or("no plan")?;

    // Every level's build, on the card after the base (the first cut: the
    // builds beside phase B are a measurement still to make).
    let first = build_level(program, family, false)?;
    let mut levels = vec![first];
    let more = |levels: &[Level]| -> bool {
        match stop {
            LevelStop::Levels(n) => levels.len() < n.clamp(1, MAX_LEVELS),
            LevelStop::Groups(g) => {
                levels.len() < MAX_LEVELS
                    && levels.last().expect("a level").build.artifacts.sizes.len() > g
            }
        }
    };
    while more(&levels) {
        let child = child_of(&plan, levels.last().expect("a level"));
        levels.push(build_level(wlfm_leg_program(&child)?, family, false)?);
    }
    times.built = now();

    // Level 1: fill and prove; then each level over the one below.
    let mut proofs: Vec<WlfmProof> = Vec::with_capacity(levels.len());
    let t = std::time::Instant::now();
    let filled = super::proof::lfm_fill_executed_under(
        &levels[0].program,
        execution,
        levels[0].build.artifacts.hasher,
        executed - execute_from,
        &Vec::new(),
    );
    let fill = t.elapsed().as_secs_f64();
    let t = std::time::Instant::now();
    proofs.push(
        super::wlfm::prove_wlfm_traces_as(&levels[0].build, filled, true, WlfmProving::Streamed)
            .map_err(|e| format!("level 1: {e}"))?,
    );
    times.levels.push(LevelClock {
        execute_from,
        executed,
        fill,
        prove: t.elapsed().as_secs_f64(),
        end: now(),
    });
    for k in 1..levels.len() {
        let execute_from = now();
        let child = child_of(&plan, &levels[k - 1]);
        let arena = wlfm_leg_arena(&child, &proofs[k - 1])?;
        let filled = super::wlfm::execute_and_fill_unmerged(
            &levels[k].program,
            &[arena],
            levels[k].build.artifacts.hasher,
        )?;
        let executed = now();
        let t = std::time::Instant::now();
        proofs.push(
            super::wlfm::prove_wlfm_traces_as(
                &levels[k].build,
                filled,
                true,
                WlfmProving::Streamed,
            )
            .map_err(|e| format!("level {}: {e}", k + 1))?,
        );
        times.levels.push(LevelClock {
            execute_from,
            executed,
            fill: 0.0,
            prove: t.elapsed().as_secs_f64(),
            end: now(),
        });
    }
    Ok(LevelsRun {
        base_proof,
        plan,
        levels,
        proofs,
        times,
    })
}
