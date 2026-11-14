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
