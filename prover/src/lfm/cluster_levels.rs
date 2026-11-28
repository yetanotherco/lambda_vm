//! Range-owned recursion. Each active machine proves its base groups in one
//! level-1 program. Level 2 verifies the leaves and closes the block bus.
//!
//! The verifier derives the leaves and join from the ELF, block statement
//! and canonical leaf map. Only the coordinator's shipment check can opt in
//! to committing the levels' prepared stacks on the card. Its only proof
//! check is the last level. The soundness ledger's per-proof convention applies to each proof;
//! the optional union bound over the chain counts M + 2 proofs.

use std::time::{Duration, Instant};

use super::{
    block_node::{BlockLayout, bind_and_publish},
    compiler::LfmProgram,
    last_level::{LastLevel, cluster_file_parts, decode_body, encode_file, last_wire},
    levels::{
        self, Level, LevelStop, build_level, check_process_format, is_last_level, level_rounds,
    },
    merkle_cap::CapMux,
    whir_block::{BlockPartition, WhirBlockPlan, block_leaf_arena, finish_program, top_claims},
    wlfm::{
        WlfmArtifacts, WlfmBuild, WlfmFamily, WlfmPrebuild, WlfmProof, WlfmProving, verify_wlfm,
    },
    wlfm_leg::{WlfmChild, emit_wlfm_leg, wlfm_leg_arena, wlfm_leg_builder, wlfm_leg_program},
};
use crate::{
    ProofOptions,
    block_whir::{
        B3Rounds, BlockFormat, BlockOptions, BlockStatement, OwnedBlockStatement,
        cluster::stages::Clock,
        cluster_tcp::{self, Topology},
    },
};
use multilinear::whir_chain::GroupFold;
use multilinear::whir_commit::Commitment;

/// Maximum fan-in of the opt-in verifier. The pinned public file reader
/// accepts only version 1, whose implicit map has one leaf.
pub const MAX_CLUSTER_LEAVES: usize = 8;

pub(crate) type LeafOwners = (Option<LeafMap>, Vec<usize>);

/// The canonical version-2 map. The byte at each group is that group's leaf,
/// numbered in order of first occurrence. Version 1 represents one leaf.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LeafMap {
    bytes: Vec<u8>,
    partition: BlockPartition,
}

impl LeafMap {
    pub fn decode(bytes: &[u8], groups: usize) -> Result<Self, String> {
        if bytes.len() != groups {
            return Err(format!(
                "leaf map length {}, expected {groups} groups",
                bytes.len()
            ));
        }
        let mut leaves: Vec<Vec<usize>> = Vec::new();
        for (group, &leaf) in bytes.iter().enumerate() {
            let leaf = usize::from(leaf);
            if leaf >= MAX_CLUSTER_LEAVES {
                return Err(format!(
                    "leaf {leaf} is past the cap of {MAX_CLUSTER_LEAVES}"
                ));
            }
            if leaf > leaves.len() {
                return Err(format!(
                    "leaf {leaf} skips leaf {} in first-group order",
                    leaves.len()
                ));
            }
            if leaf == leaves.len() {
                leaves.push(Vec::new());
            }
            leaves[leaf].push(group);
        }
        if leaves.len() < 2 {
            return Err("one leaf uses version 1 without a leaf map".into());
        }
        Ok(Self {
            bytes: bytes.to_vec(),
            partition: BlockPartition::new(leaves, groups)?,
        })
    }

    pub fn as_bytes(&self) -> &[u8] {
        &self.bytes
    }

    pub(crate) fn partition(&self, groups: usize) -> Result<BlockPartition, String> {
        if self.partition.num_units() != groups {
            return Err("leaf map length disagrees with the statement".into());
        }
        Ok(self.partition.clone())
    }

    /// Owner indices are private to the runner. The file numbers active owners
    /// by their first group, including when a range closes with no groups.
    pub(crate) fn from_owners(owners: &[usize], machines: usize) -> Result<LeafOwners, String> {
        if machines == 0 || machines > MAX_CLUSTER_LEAVES || owners.is_empty() {
            return Err(format!(
                "cluster needs groups and at most {MAX_CLUSTER_LEAVES} machines"
            ));
        }
        let mut order = Vec::new();
        let mut map = Vec::with_capacity(owners.len());
        for &owner in owners {
            if owner >= machines {
                return Err("group owner is outside the cluster".into());
            }
            let leaf = if let Some(leaf) = order.iter().position(|&id| id == owner) {
                leaf
            } else {
                let leaf = order.len();
                order.push(owner);
                leaf
            };
            map.push(leaf as u8);
        }
        let map = (order.len() > 1)
            .then(|| Self::decode(&map, owners.len()))
            .transpose()?;
        Ok((map, order))
    }
}

pub(crate) fn check_cluster_family(family: WlfmFamily) -> Result<(), String> {
    if family != WlfmFamily::B3 {
        return Err("range-owned levels support only BLAKE3".into());
    }
    Ok(())
}

fn check_leaf_stop(stop: LevelStop, level_one_groups: usize) -> Result<(), String> {
    if is_last_level(stop, 1, level_one_groups) {
        return Err("a chain with several leaves cannot stop at level 1".into());
    }
    Ok(())
}

fn pinned_format(family: WlfmFamily) -> Result<BlockFormat, String> {
    levels::pin_family(
        BlockFormat::pinned_with(GroupFold::Indexed),
        family,
        B3Rounds::Fri,
    )
}

pub(crate) fn cluster_plan(
    elf: &[u8],
    options: &ProofOptions,
    format: &BlockFormat,
    statement: BlockStatement<'_>,
    prepared: Option<&[(usize, Vec<Commitment>)]>,
    map: Option<&LeafMap>,
) -> Result<WhirBlockPlan, String> {
    crate::base_only::refuse_recursion()?;
    let plan = WhirBlockPlan::derive_blake3_with_share(
        elf,
        options,
        format,
        statement,
        1,
        prepared,
        false,
        true,
        Some(CapMux::Packed),
    )?;
    match map {
        Some(map) => plan.with_cluster_leaf_map(map),
        None => Ok(plan),
    }
}

pub(crate) fn leaf_child(plan: &WhirBlockPlan, build: &WlfmBuild) -> WlfmChild {
    WlfmChild {
        artifacts: build.artifacts.clone(),
        num_public_words: plan.child_layout().total(),
        cap_mux: Some(CapMux::Packed),
    }
}

fn emit_join(
    children: &[WlfmChild],
    layout: &BlockLayout,
    bind: impl FnOnce(
        &mut super::builder::LfmBuilder,
        &[super::per_table_aggregator::LegCells],
        &BlockLayout,
    ),
) -> Result<LfmProgram, String> {
    if !(2..=MAX_CLUSTER_LEAVES).contains(&children.len())
        || layout.state_words != 2
        || !layout.has_sum
    {
        return Err("the join requires several BLAKE3 leaves with bus shares".into());
    }
    let mut b = wlfm_leg_builder(WlfmFamily::B3).with_cap_mux(CapMux::Packed);
    let mut legs = Vec::with_capacity(children.len());
    for child in children {
        if child.num_public_words != layout.total() {
            return Err("the join's leaf word count disagrees with the child layout".into());
        }
        let arena = b.declare_arena(0);
        legs.push(emit_wlfm_leg(&mut b, child, arena)?);
    }
    bind(&mut b, &legs, layout);
    finish_program(b)
}

fn join_program(children: &[WlfmChild], layout: &BlockLayout) -> Result<LfmProgram, String> {
    emit_join(children, layout, |b, legs, layout| {
        bind_and_publish(b, legs, layout, true)
    })
}

/// A group-count stop reads the sum of the leaves' W-LFM group counts at
/// level 1 and the single proof's group count at each higher level.
fn join_level(
    children: &[WlfmChild],
    layout: &BlockLayout,
    stop: LevelStop,
    last: LastLevel,
    rounds: Option<u8>,
    on_host: bool,
) -> Result<Level, String> {
    check_leaf_stop(stop, children.iter().map(|c| c.artifacts.sizes.len()).sum())?;
    build_level(
        join_program(children, layout)?,
        WlfmFamily::B3,
        rounds,
        on_host,
        (stop, 2),
        last,
    )
}

fn build_leaf(plan: &WhirBlockPlan, leaf: usize, on_host: bool) -> Result<Level, String> {
    build_leaf_program(plan.leaf_program(leaf)?, level_rounds(plan), on_host)
}

fn build_leaf_program(
    program: LfmProgram,
    rounds: Option<u8>,
    on_host: bool,
) -> Result<Level, String> {
    build_level(
        program,
        WlfmFamily::B3,
        rounds,
        on_host,
        (LevelStop::DEFAULT, 1),
        LastLevel::pinned(WlfmFamily::B3),
    )
}

#[cfg(all(test, feature = "test-security"))]
fn prove_level(level: &Level, arena: &[Vec<super::LfmWord>]) -> Result<WlfmProof, String> {
    prove_level_staged(level, arena, Clock::default(), "leaf")
}

fn prove_level_staged(
    level: &Level,
    arena: &[Vec<super::LfmWord>],
    clock: Clock,
    stage: &str,
) -> Result<WlfmProof, String> {
    let filled = super::wlfm::execute_and_fill_unmerged(
        &level.program,
        arena,
        level.build.artifacts.hasher,
    )?;
    let (execute, fill) = filled.execute_and_fill_secs();
    clock.mark_with(
        format_args!("{stage}-fill-done"),
        format_args!(" execute={execute:.6} fill={fill:.6}"),
    );
    clock.mark(format_args!("{stage}-prove-start"));
    let proof =
        super::wlfm::prove_wlfm_traces_as(&level.build, filled, true, WlfmProving::Streamed)?;
    if let Some(stamps) = super::wlfm::take_wlfm_stamps() {
        let pool_ready = multilinear::pinned::ready();
        let pool_requested = multilinear::pinned::stats().is_some();
        clock.mark_with(
            format_args!("{stage}-prove-end"),
            format_args!(
                " prep={:.6} phase_a={:.6} phase_b={:.6} record_tables={} record_bytes={} host_tables={} host_bytes={} pool_requested={pool_requested} pool_ready={pool_ready}",
                stamps.prep, stamps.phase_a, stamps.phase_b,
                stamps.records_tables, stamps.records_bytes, stamps.host_tables, stamps.host_bytes,
            ),
        );
    }
    Ok(proof)
}

pub(crate) struct LeafPrebuild {
    program: LfmProgram,
    pre: WlfmPrebuild,
}

impl LeafPrebuild {
    fn new(
        program: LfmProgram,
        rounds: Option<u8>,
        last: LastLevel,
        clock: Clock,
    ) -> Result<Self, String> {
        let pre = levels::prepare_level_program(
            &program,
            WlfmFamily::B3,
            rounds,
            (LevelStop::DEFAULT, 1),
            last,
        )?;
        clock.mark("leaf-prebuild-done");
        Ok(Self { program, pre })
    }

    fn prove<T>(
        self,
        leaf: usize,
        clock: Clock,
        prove: impl FnOnce(&Level) -> Result<T, String>,
    ) -> Result<T, String> {
        levels::with_range_leaf_memory(leaf, || {
            let build = levels::commit_level_staged(self.pre, 1, clock)?;
            clock.mark("leaf-build-done");
            prove(&Level {
                program: self.program,
                build,
            })
        })
    }
}

pub(crate) fn prebuild_leaf(
    plan: &WhirBlockPlan,
    leaf: usize,
    closed: bool,
    clock: Clock,
) -> Result<LeafPrebuild, String> {
    let program = if closed {
        plan.closed_block_program_staged(clock)?
    } else {
        plan.leaf_program_marked_staged(leaf, clock)?.0
    };
    LeafPrebuild::new(
        program,
        level_rounds(plan),
        LastLevel::pinned(WlfmFamily::B3),
        clock,
    )
}

#[cfg(all(test, feature = "test-security"))]
pub(crate) fn prove_leaf(
    plan: &WhirBlockPlan,
    leaf: usize,
    arena: &[Vec<super::LfmWord>],
) -> Result<(WlfmChild, WlfmProof), String> {
    prove_leaf_staged(plan, leaf, arena, Clock::default())
}

pub(crate) fn prove_leaf_staged(
    plan: &WhirBlockPlan,
    leaf: usize,
    arena: &[Vec<super::LfmWord>],
    clock: Clock,
) -> Result<(WlfmChild, WlfmProof), String> {
    let pre = prebuild_leaf(plan, leaf, false, clock)?;
    prove_prebuilt_leaf(plan, leaf, pre, arena, clock)
}

pub(crate) fn prove_prebuilt_leaf(
    plan: &WhirBlockPlan,
    leaf: usize,
    pre: LeafPrebuild,
    arena: &[Vec<super::LfmWord>],
    clock: Clock,
) -> Result<(WlfmChild, WlfmProof), String> {
    prove_leaf_with_artifacts(plan, leaf, pre, arena, clock, |_| Ok(()))
}

pub(crate) fn prove_leaf_with_artifacts(
    plan: &WhirBlockPlan,
    leaf: usize,
    pre: LeafPrebuild,
    arena: &[Vec<super::LfmWord>],
    clock: Clock,
    built: impl FnOnce(&WlfmChild) -> Result<(), String>,
) -> Result<(WlfmChild, WlfmProof), String> {
    pre.prove(leaf, clock, |level| {
        let child = leaf_child(plan, &level.build);
        built(&child)?;
        Ok((child, prove_level_staged(level, arena, clock, "leaf")?))
    })
}

#[cfg(test)]
pub(crate) fn prebuild_weakened_leaf(
    plan: &WhirBlockPlan,
    leaf: usize,
    clock: Clock,
) -> Result<LeafPrebuild, String> {
    use super::whir_block::{LeafChecks, leaf_program_with};
    let program = leaf_program_with(
        plan,
        leaf,
        LeafChecks {
            opening: false,
            ..LeafChecks::ALL
        },
    )?;
    LeafPrebuild::new(
        program,
        level_rounds(plan),
        LastLevel::pinned(WlfmFamily::B3),
        clock,
    )
}

#[cfg(all(test, feature = "test-security"))]
pub(crate) fn prove_single(
    plan: &WhirBlockPlan,
    arena: &[Vec<super::LfmWord>],
    last: LastLevel,
) -> Result<WlfmProof, String> {
    prove_single_staged(plan, arena, last, Clock::default())
}

pub(crate) fn prove_single_staged(
    plan: &WhirBlockPlan,
    arena: &[Vec<super::LfmWord>],
    last: LastLevel,
    clock: Clock,
) -> Result<WlfmProof, String> {
    let pre = LeafPrebuild::new(
        plan.closed_block_program_staged(clock)?,
        level_rounds(plan),
        last,
        clock,
    )?;
    prove_prebuilt_single(plan, pre, arena, last, clock)
}

pub(crate) fn prove_prebuilt_single(
    plan: &WhirBlockPlan,
    pre: LeafPrebuild,
    arena: &[Vec<super::LfmWord>],
    last: LastLevel,
    clock: Clock,
) -> Result<WlfmProof, String> {
    let (child, proof) = pre.prove(0, clock, |level| {
        Ok((
            levels::child_of(plan, level),
            prove_level_staged(level, arena, clock, "leaf")?,
        ))
    })?;
    let arena = wlfm_leg_arena(&child, &proof)?;
    prove_join_program(
        || wlfm_leg_program(&child),
        &[arena],
        WlfmFamily::B3,
        level_rounds(plan),
        last,
        clock,
    )
}

#[cfg(all(test, feature = "test-security"))]
pub(crate) fn prove_join(
    layout: &BlockLayout,
    rounds: Option<u8>,
    leaves: Vec<(WlfmChild, WlfmProof)>,
    last: LastLevel,
) -> Result<WlfmProof, String> {
    prove_join_staged(layout, rounds, leaves, last, Clock::default())
}

pub(crate) fn prove_join_staged(
    layout: &BlockLayout,
    rounds: Option<u8>,
    leaves: Vec<(WlfmChild, WlfmProof)>,
    last: LastLevel,
    clock: Clock,
) -> Result<WlfmProof, String> {
    let (children, proofs): (Vec<_>, Vec<_>) = leaves.into_iter().unzip();
    let arenas = join_arenas(&children, proofs)?;
    let level = build_join(&children, layout, rounds, last, clock)?;
    prove_level_staged(&level, &arenas, clock, "join")
}

pub(crate) fn build_join(
    children: &[WlfmChild],
    layout: &BlockLayout,
    rounds: Option<u8>,
    last: LastLevel,
    clock: Clock,
) -> Result<Level, String> {
    check_leaf_stop(
        LevelStop::DEFAULT,
        children.iter().map(|c| c.artifacts.sizes.len()).sum(),
    )?;
    build_join_program(
        || join_program(children, layout),
        WlfmFamily::B3,
        rounds,
        last,
        clock,
    )
}

fn join_arenas(
    children: &[WlfmChild],
    proofs: Vec<WlfmProof>,
) -> Result<Vec<Vec<super::LfmWord>>, String> {
    if children.len() != proofs.len() {
        return Err("the join requires one proof for each leaf".into());
    }
    children
        .iter()
        .zip(&proofs)
        .map(|(child, proof)| wlfm_leg_arena(child, proof))
        .collect()
}

pub(crate) fn prove_built_join(
    level: Level,
    children: &[WlfmChild],
    proofs: Vec<WlfmProof>,
    clock: Clock,
) -> Result<WlfmProof, String> {
    let arenas = join_arenas(children, proofs)?;
    prove_level_staged(&level, &arenas, clock, "join")
}

fn prove_join_program(
    program: impl FnOnce() -> Result<LfmProgram, String>,
    arenas: &[Vec<super::LfmWord>],
    family: WlfmFamily,
    rounds: Option<u8>,
    last: LastLevel,
    clock: Clock,
) -> Result<WlfmProof, String> {
    let level = build_join_program(program, family, rounds, last, clock)?;
    prove_level_staged(&level, arenas, clock, "join")
}

fn build_join_program(
    program: impl FnOnce() -> Result<LfmProgram, String>,
    family: WlfmFamily,
    rounds: Option<u8>,
    last: LastLevel,
    clock: Clock,
) -> Result<Level, String> {
    clock.mark("join-compile-start");
    let program = program()?;
    clock.mark("join-compile-end");
    clock.mark("join-build-start");
    let level = build_level(
        program,
        family,
        rounds,
        false,
        (LevelStop::DEFAULT, 2),
        last,
    )?;
    clock.mark("join-build-end");
    Ok(level)
}

/// Verify a version-2 file at the pinned BLAKE3 format. Version 1 is
/// still read by the one-leaf verifier.
pub fn verify_cluster_block_levels_file(
    elf: &[u8],
    statement: BlockStatement<'_>,
    family: WlfmFamily,
    file: &[u8],
) -> Result<(), String> {
    verify_cluster_block_levels_file_at(
        elf,
        statement,
        family,
        LevelStop::DEFAULT,
        LastLevel::pinned(family),
        file,
    )
}

/// Explicit choice of chain stop and last-level format. A version-1 file
/// goes to the one-leaf verifier under both choices. A version-2 file takes
/// only the default stop, the one its provers make. The map is validated and
/// the proof decoded before any program is derived. Every program is then
/// derived from the ELF, statement and typed map, with its prepared stacks on
/// the host.
pub fn verify_cluster_block_levels_file_at(
    elf: &[u8],
    statement: BlockStatement<'_>,
    family: WlfmFamily,
    stop: LevelStop,
    last: LastLevel,
    file: &[u8],
) -> Result<(), String> {
    check_cluster_file(elf, statement, family, stop, last, file, FileCheck::Public).map(|_| ())
}

enum FileCheck<'a> {
    Public,
    Shipment {
        map: Option<&'a LeafMap>,
        on_host: bool,
    },
}

/// Check the exact encoded file and return the time spent deriving programs.
fn check_cluster_file(
    elf: &[u8],
    statement: BlockStatement<'_>,
    family: WlfmFamily,
    stop: LevelStop,
    last: LastLevel,
    file: &[u8],
    check: FileCheck<'_>,
) -> Result<Duration, String> {
    let (map, wire, body) = cluster_file_parts(file, last, statement.groups.len())?;
    let (map, on_host) = match check {
        FileCheck::Public => (map.as_ref(), true),
        FileCheck::Shipment {
            map: expected,
            on_host,
        } => {
            if map.as_ref() != expected {
                return Err("the file's leaf map disagrees with the coordinator's map".into());
            }
            (expected, on_host)
        }
    };
    let Some(map) = map else {
        let format = pinned_format(family)?;
        let proof = decode_body(wire, body)?;
        check_process_format(family, last)?;
        levels::check_last_level_family(last, family)?;
        let start = Instant::now();
        let plan = levels::verifier_level_plan(elf, &format, statement, family, CapMux::Packed)?;
        let levels = levels::derive_levels_last(&plan, family, stop, last, on_host)?;
        let top = levels.last().expect("a level");
        let derivation = start.elapsed();
        levels::verify_derived_level(&plan, last, &proof, &top.build.artifacts)?;
        return Ok(derivation);
    };
    check_cluster_family(family)?;
    if stop != LevelStop::DEFAULT {
        return Err(format!(
            "a file with several leaves takes only the default chain stop ({:?}); refused {stop:?}",
            LevelStop::DEFAULT
        ));
    }
    check_process_format(family, last)?;
    levels::check_last_level_family(last, family)?;
    let proof = decode_body(wire, body)?;
    let start = Instant::now();
    let format = pinned_format(family)?;
    let options = super::proof::block_base_options_under(&format.zf);
    let (plan, _, join) = derive_join(elf, &options, &format, statement, map, last, on_host)?;
    let derivation = start.elapsed();
    verify_wlfm(&join, &proof, &proof.public_words)
        .map_err(|e| format!("the cluster last level does not verify: {e}"))?;
    if !top_claims(&plan, &proof.public_words) {
        return Err("the last level does not claim this block".into());
    }
    Ok(derivation)
}

#[cfg(test)]
thread_local! {
    /// Join chains this thread derived for the verifier.
    pub(crate) static DERIVATIONS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

/// The shared plan and the leaf and join artifacts derived from it. Only the
/// level stacks use the selected placement. The base plan independently
/// commits its own prepared columns from the ELF, statement and pinned format.
fn derive_join(
    elf: &[u8],
    options: &ProofOptions,
    format: &BlockFormat,
    statement: BlockStatement<'_>,
    map: &LeafMap,
    last: LastLevel,
    on_host: bool,
) -> Result<(WhirBlockPlan, Vec<WlfmChild>, WlfmArtifacts), String> {
    #[cfg(test)]
    DERIVATIONS.with(|count| count.set(count.get() + 1));
    let plan = cluster_plan(elf, options, format, statement, None, Some(map))?;
    let children = (0..plan.partition().num_leaves())
        .map(|i| build_leaf(&plan, i, on_host).map(|level| leaf_child(&plan, &level.build)))
        .collect::<Result<Vec<_>, _>>()?;
    let join = join_level(
        &children,
        &plan.child_layout(),
        LevelStop::DEFAULT,
        last,
        level_rounds(&plan),
        on_host,
    )?;
    Ok((plan, children, join.build.artifacts))
}

#[cfg(test)]
thread_local! {
    /// Shipments this thread checked with the levels verifier.
    pub(crate) static FILE_CHECKS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
    /// The errors this thread's range attempts fell back on, oldest first.
    pub(crate) static FALLBACK_ERRORS: std::cell::RefCell<Vec<String>> =
        const { std::cell::RefCell::new(Vec::new()) };
}

/// A statement and its verified last-level file, ready for publication.
pub struct VerifiedClusterFile {
    statement: OwnedBlockStatement,
    file: Vec<u8>,
    clock: Clock,
}

impl VerifiedClusterFile {
    #[doc(hidden)]
    pub fn stage_clock(&self) -> Clock {
        self.clock
    }
    pub fn into_parts(self) -> (OwnedBlockStatement, Vec<u8>) {
        (self.statement, self.file)
    }
}

/// The encoded proof and statement awaiting the coordinator's check.
/// Only [`check_shipment`] can promote it to [`VerifiedClusterFile`]:
///
/// ```
/// use lambda_vm_prover::lfm::cluster_levels::{
///     UncheckedClusterFile, VerifiedClusterFile, check_shipment,
/// };
///
/// fn as_verified(file: UncheckedClusterFile<'_>) -> Result<VerifiedClusterFile, String> {
///     check_shipment(file)
/// }
/// ```
///
/// The same function with a conversion in place of the check does not
/// compile (E0277: no `From` or `Into` between the two). The two examples
/// differ only in that line, so the one above shows the imports and the
/// signature are not what fails here.
///
/// ```compile_fail
/// use lambda_vm_prover::lfm::cluster_levels::{
///     UncheckedClusterFile, VerifiedClusterFile, check_shipment,
/// };
///
/// fn as_verified(file: UncheckedClusterFile<'_>) -> Result<VerifiedClusterFile, String> {
///     Ok(file.into())
/// }
/// ```
pub struct UncheckedClusterFile<'elf> {
    elf: &'elf [u8],
    contents: (OwnedBlockStatement, Vec<u8>),
    check: ShipmentCheck,
    clock: Clock,
}

enum ShipmentCheck {
    Replica(WlfmFamily),
    Range(Option<LeafMap>),
}

impl<'elf> UncheckedClusterFile<'elf> {
    fn encode(
        elf: &'elf [u8],
        statement: OwnedBlockStatement,
        proof: &WlfmProof,
        check: ShipmentCheck,
        clock: Clock,
    ) -> Result<Self, String> {
        let (family, map) = match &check {
            ShipmentCheck::Replica(family) => (*family, None),
            ShipmentCheck::Range(map) => (WlfmFamily::B3, map.as_ref()),
        };
        let file = encode_file(proof, LastLevel::pinned(family), last_wire()?, map)?;
        clock.mark("inner-encode-done");
        Ok(Self {
            elf,
            contents: (statement, file),
            check,
            clock,
        })
    }

    #[doc(hidden)]
    pub fn stage_clock(&self) -> Clock {
        self.clock
    }

    /// The statement and exact proof bytes to serialize for publication.
    pub fn contents(&self) -> &(OwnedBlockStatement, Vec<u8>) {
        &self.contents
    }
}

/// Check the shipment's encoded proof using the coordinator's own inputs.
/// Range checks use host derivation by default and can opt into card
/// derivation with `LAMBDA_VM_CLUSTER_CHECK_ON_CARD=1`.
pub fn check_shipment(shipment: UncheckedClusterFile<'_>) -> Result<VerifiedClusterFile, String> {
    let UncheckedClusterFile {
        elf,
        contents: (statement, file),
        check,
        clock,
    } = shipment;
    match check {
        ShipmentCheck::Replica(family) => {
            levels::verify_block_levels_file(elf, statement.view(), family, &file)?;
        }
        ShipmentCheck::Range(map) => {
            check_shipment_file(elf, statement.view(), map.as_ref(), &file)?;
        }
    }
    #[cfg(test)]
    FILE_CHECKS.with(|checks| checks.set(checks.get() + 1));
    Ok(VerifiedClusterFile {
        statement,
        file,
        clock,
    })
}

fn check_shipment_file(
    elf: &[u8],
    statement: BlockStatement<'_>,
    map: Option<&LeafMap>,
    file: &[u8],
) -> Result<(), String> {
    let family = WlfmFamily::B3;
    let on_host = std::env::var("LAMBDA_VM_CLUSTER_CHECK_ON_CARD").as_deref() != Ok("1");
    let start = Instant::now();
    let derivation = check_cluster_file(
        elf,
        statement,
        family,
        LevelStop::DEFAULT,
        LastLevel::pinned(family),
        file,
        FileCheck::Shipment { map, on_host },
    )?;
    let verify = start.elapsed() - derivation;
    eprintln!(
        "CLUSTER LEVELS FILE VERIFY derivation={:.6}s verify={:.6}s",
        derivation.as_secs_f64(),
        verify.as_secs_f64()
    );
    Ok(())
}

#[cfg(all(test, feature = "test-security"))]
fn verify_shipment(
    elf: &[u8],
    statement: OwnedBlockStatement,
    proof: &WlfmProof,
    map: Option<&LeafMap>,
) -> Result<VerifiedClusterFile, String> {
    check_before_return(UncheckedClusterFile::encode(
        elf,
        statement,
        proof,
        ShipmentCheck::Range(map.cloned()),
        Clock::default(),
    )?)
}

fn check_before_return(shipment: UncheckedClusterFile<'_>) -> Result<VerifiedClusterFile, String> {
    let clock = shipment.stage_clock();
    clock.mark("file-check-start");
    let verified = check_shipment(shipment)
        .map_err(|e| format!("cluster file verification before publication: {e}"))?;
    clock.mark("file-check-end");
    Ok(verified)
}

/// Prove one level-1 program per active range machine and verify the exact
/// file returned by this entry point.
pub fn prove_cluster_block_levels(
    elf: &[u8],
    input: &[u8],
    options: &BlockOptions,
    family: WlfmFamily,
    topology: &Topology,
) -> Result<VerifiedClusterFile, String> {
    check_before_return(prove_cluster_block_levels_unchecked(
        elf, input, options, family, topology,
    )?)
}

/// Assemble a range-owned shipment for publication followed by a check.
pub fn prove_cluster_block_levels_unchecked<'elf>(
    elf: &'elf [u8],
    input: &[u8],
    options: &BlockOptions,
    family: WlfmFamily,
    topology: &Topology,
) -> Result<UncheckedClusterFile<'elf>, String> {
    check_cluster_family(family)?;
    check_process_format(family, LastLevel::pinned(family))?;
    let format = pinned_format(family)?;
    let opts = super::proof::block_base_options_under(&format.zf);
    crate::base_only::refuse_recursion()?;
    let ((statement, map, proof), clock) = match cluster_tcp::prove_range_levels(
        topology, elf, input, &opts, &format, options,
    ) {
        Ok(result) => result,
        Err(error) if topology.local_fallback() => {
            let cuts = topology
                .range_cuts()
                .ok_or("range levels require a windows list")?;
            let count = cluster_tcp::count_local_fallback();
            eprintln!(
                "CLUSTER TCP: {error}; starting a fresh range reference attempt (fallback {count})"
            );
            #[cfg(test)]
            FALLBACK_ERRORS.with(|errors| errors.borrow_mut().push(error.clone()));
            return prove_range_levels_reference_unchecked(elf, input, options, family, cuts);
        }
        Err(error) => return Err(error),
    };
    UncheckedClusterFile::encode(elf, statement, &proof, ShipmentCheck::Range(map), clock)
}

/// Assemble a one-leaf shipment with TCP replica proving or, with no topology,
/// a fresh local run using the same fallback placement.
pub fn prove_replica_levels_unchecked<'elf>(
    elf: &'elf [u8],
    input: &[u8],
    options: &BlockOptions,
    family: WlfmFamily,
    topology: Option<&Topology>,
) -> Result<UncheckedClusterFile<'elf>, String> {
    let format = pinned_format(family)?;
    let opts = super::proof::block_base_options_under(&format.zf);
    let prove = || {
        levels::prove_block_levels(
            elf,
            input,
            &opts,
            &format,
            options,
            family,
            LevelStop::DEFAULT,
            true,
            true,
        )
    };
    let run = match topology {
        Some(topology) => cluster_tcp::coordinate(topology, prove)?,
        None => cluster_tcp::locally(prove)?,
    };
    let last = run
        .proofs
        .last()
        .ok_or("the levels produced no final proof")?;
    UncheckedClusterFile::encode(
        elf,
        run.base_proof.statement().to_owned(),
        last,
        ShipmentCheck::Replica(family),
        Clock::default(),
    )
}

/// One-process reference for a range list. Its map is formed by numbering
/// the same range owners in first-group order. Use the same fixed trace seed
/// and deterministic grinding on both runs to compare file bytes.
pub fn prove_range_levels_reference(
    elf: &[u8],
    input: &[u8],
    options: &BlockOptions,
    family: WlfmFamily,
    cuts: &[usize],
) -> Result<VerifiedClusterFile, String> {
    check_before_return(prove_range_levels_reference_unchecked(
        elf, input, options, family, cuts,
    )?)
}

/// [`prove_range_levels_reference`] without its check, for a caller that
/// checks the shipment itself with [`check_shipment`].
pub fn prove_range_levels_reference_unchecked<'elf>(
    elf: &'elf [u8],
    input: &[u8],
    options: &BlockOptions,
    family: WlfmFamily,
    cuts: &[usize],
) -> Result<UncheckedClusterFile<'elf>, String> {
    crate::base_only::refuse_recursion()?;
    check_cluster_family(family)?;
    check_process_format(family, LastLevel::pinned(family))?;
    let format = pinned_format(family)?;
    let opts = super::proof::block_base_options_under(&format.zf);
    if cuts.is_empty() {
        let run = levels::prove_block_levels(
            elf,
            input,
            &opts,
            &format,
            options,
            family,
            LevelStop::DEFAULT,
            true,
            true,
        )?;
        return UncheckedClusterFile::encode(
            elf,
            run.base_proof.statement().to_owned(),
            &run.proofs[1],
            ShipmentCheck::Range(None),
            Clock::default(),
        );
    }
    let clock = Clock::reference();
    let (base, prepared, (map, owners)) =
        crate::block_whir::prove_range_reference(elf, input, &opts, &format, options, cuts, clock)
            .map_err(|e| e.to_string())?;
    let statement = base.statement();
    let plan = cluster_plan(
        elf,
        &opts,
        &format,
        statement,
        Some(&prepared),
        map.as_ref(),
    )?;
    clock.mark("leaf-plan-done");
    let proof = if map.is_some() {
        let leaves = owners
            .into_iter()
            .enumerate()
            .map(|(leaf, owner)| {
                let lane = if owner == cuts.len() {
                    clock
                } else {
                    clock.for_worker(owner)
                };
                lane.mark("reference-leaf-start");
                let arena = block_leaf_arena(&plan, &base, leaf)?;
                lane.mark("leaf-arena-done");
                prove_leaf_staged(&plan, leaf, &arena, lane)
            })
            .collect::<Result<Vec<_>, _>>()?;
        prove_join_staged(
            &plan.child_layout(),
            level_rounds(&plan),
            leaves,
            LastLevel::pinned(family),
            clock,
        )?
    } else {
        let arena = block_leaf_arena(&plan, &base, 0)?;
        clock.mark("leaf-arena-done");
        prove_single_staged(&plan, &arena, LastLevel::pinned(family), clock)?
    };
    UncheckedClusterFile::encode(
        elf,
        statement.to_owned(),
        &proof,
        ShipmentCheck::Range(map),
        clock,
    )
}

#[cfg(all(test, feature = "cuda"))]
fn assert_card_artifacts_match_host(elf: &[u8], statement: BlockStatement<'_>, map: &LeafMap) {
    let format = pinned_format(WlfmFamily::B3).unwrap();
    let last = LastLevel::pinned(WlfmFamily::B3);
    let opts = crate::lfm::proof::block_base_options_under(&format.zf);
    let derive = |on_host| {
        let (plan, leaves, join) =
            derive_join(elf, &opts, &format, statement, map, last, on_host).unwrap();
        let base_roots = plan
            .prepared()
            .iter()
            .map(|stack| stack.roots.clone())
            .collect::<Vec<_>>();
        let levels = leaves
            .into_iter()
            .map(|leaf| leaf.artifacts)
            .chain([join])
            .map(|a| (a.program_id, a.derived_roots()))
            .collect::<Vec<_>>();
        (base_roots, levels)
    };
    let (used, _, _, _) = multilinear::gpu::settle_card().expect("CUDA device");
    let card = derive(false);
    let (_, peak, _, _) = multilinear::gpu::settle_card().expect("CUDA device");
    assert!(peak > used, "derivation did not allocate on the card");
    assert!(card.1.len() > 2);
    assert!(
        card.1[..card.1.len() - 1]
            .iter()
            .all(|(_, roots)| !roots.is_empty())
    );
    assert_eq!(derive(true), card);
}

/// Pins that must run under the deterministic settings rerun themselves in a
/// child of the test binary.
#[cfg(test)]
pub(crate) mod pin_process {
    const FIXED: [&str; 2] = [
        "LAMBDA_VM_DETERMINISTIC_GRIND",
        "LAMBDA_VM_FIXED_TRACE_HASH",
    ];

    /// The SHA-256 of the fixture ELF (`test_commit_split`) the pins were
    /// recorded with, as clang 18.1.3 assembles it. Another assembler lays the
    /// program out differently, and every pinned digest and id moves with it.
    const PINNED_FIXTURE_ELF: &str =
        "d80356e2357e742b7fb0d367ca53b464e7c2b2d5083af2b6d8f03f07338098fc";

    /// Refuses a fixture ELF other than the one the pins were recorded with,
    /// by name, before any pinned value is compared.
    pub(crate) fn check_pinned_elf(elf: &[u8]) {
        use digest::Digest;
        let digest = format!("{:x}", sha2::Sha256::digest(elf));
        assert_eq!(
            digest, PINNED_FIXTURE_ELF,
            "the fixture ELF is not the one the pins were recorded with (another assembler?)"
        );
    }

    /// Whether this process already runs under the deterministic settings.
    pub(crate) fn deterministic() -> bool {
        FIXED
            .iter()
            .all(|key| std::env::var(key).as_deref() == Ok("1"))
    }

    /// Whether a libtest run reports exactly one test, passed.
    pub(crate) fn one_test_passed(stdout: &str) -> bool {
        stdout
            .lines()
            .any(|line| line.starts_with("test result: ok. 1 passed; 0 failed;"))
    }

    /// Runs `test` (ignored or not) alone in a child of this test binary
    /// under the deterministic settings and the dedup last-level wire. Panics
    /// unless that one test ran and passed.
    pub(crate) fn run_alone(test: &str) {
        let mut child = std::process::Command::new(std::env::current_exe().unwrap());
        child.args(["--exact", test, "--include-ignored", "--test-threads=1"]);
        for key in FIXED {
            child.env(key, "1");
        }
        let result = child
            .env("LAMBDA_VM_LAST_LEVEL_WIRE", "dedup")
            .env_remove("LAMBDA_VM_CLUSTER")
            .env_remove("LAMBDA_VM_CLUSTER_WINDOWS")
            .output()
            .unwrap();
        let stdout = String::from_utf8_lossy(&result.stdout);
        assert!(
            result.status.success() && one_test_passed(&stdout),
            "{test} in a child:\n{stdout}\n{}",
            String::from_utf8_lossy(&result.stderr)
        );
    }
}

#[cfg(all(test, feature = "test-security"))]
#[path = "cluster_levels_tests.rs"]
mod tests;

#[cfg(all(test, not(feature = "test-security")))]
#[path = "cluster_levels_pin_tests.rs"]
mod production_pin;
