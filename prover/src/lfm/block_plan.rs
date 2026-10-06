//! The no-epoch block tree's PLAN: every leaf, node and top program of a
//! block's recursion, derived by the verifier from the ELF it trusts, the proof
//! options and the block's shape — never from a proof.
//!
//! # Why a plan
//!
//! A leaf's program is a function of its instances' shapes: the part count, the
//! OOD blocks, whether an instance carries an aux root and a bus contribution,
//! its FRI shape. Read from a prover's proof, those are the prover's choice: an
//! inflated part count widens the composition polynomial's degree space, a
//! dropped contribution drops `L` from the transcript and the bus sum. The host
//! verifier pins every one of them to the AIR (`verifier.rs:1730`, `:1742`,
//! `:1808-1818`); a consumer that emitted the tree from proof-read shapes would
//! verify against weaker programs than the host's. So the plan takes each shape
//! from the AIR at the claimed trace length
//! ([`TableChallengeShape::derive`], [`TableVerifyShape::derive`]), and the
//! statement's fields through the host's own pre-checks
//! (`verify_proof_parts`): the accelerator shape, the private-page bound, the
//! page configs, the sub-proof count.
//!
//! # What the plan fixes
//!
//! - The partition: [`partition_by_rule`] over the instances' closed-form costs
//!   (D-NOEPOCH §12.2's rule) — a pure function of the shape, no env read.
//! - The carrier: leaf 0 subtracts the COMMIT-bus target, and no other leaf can
//!   (the leaf emitter computes `carries = leaf == plan.carrier()`).
//! - The attestation id a leaf publishes: computed from the very preprocessed
//!   roots the leaf absorbs (DECODE's, the ELF data pages').
//!
//! # The block verifier
//!
//! [`verify_block_tree`]: derive the plan, derive the top program by emitting
//! every leaf and node and building their artifacts (no proof), verify the top
//! proof against THOSE artifacts, and compare the published id and output with
//! the trusted ELF's id and the claimed output. A tree over any other programs —
//! another partition, another carrier, a weakened shape — has another top, and
//! its proof is refused there.

use std::time::Instant;

use stark::config::Commitment;

use crate::tables::types::FE;

use super::block_leaf::{BlockPartition, partition_by_rule};
use super::block_node::{BlockLayout, BlockNodeInputs};
use super::builder::LfmBuilder;
use super::compiler::{LfmProgram, compile};
use super::constraints::Analysis;
use super::epoch::TableChallengeShape;
use super::epoch_verify::TableVerifyShape;
use super::per_table_aggregator::DerivedChild;
use super::programs::AttestedInputs;
use super::registry::LfmArtifacts;
use super::word::{LfmWord, base_word};

/// Permutations of transcript and grinding per instance fork, on top of its
/// legs' closed form: D-NOEPOCH §12.2's 694 (690 transcript + 4 PoW, the mean
/// over 337 measured sub-proofs).
pub const FORK_PERMS: usize = 694;

/// The leaf-load cap: today's wrap LFM_HASH shape, 2^18 + 2^15 rows, less the
/// headroom wrap 2 runs at (D-NOEPOCH §12.2).
pub const LEAF_PERMS_CAP: usize = 279_000;

/// The partition's cost model, versioned: per instance the legs' closed form
/// (`table_permutations_for` under the production wrap hash) plus
/// [`FORK_PERMS`], leaves filled under [`LEAF_PERMS_CAP`] by
/// [`partition_by_rule`]. The partition — and so every leaf's program and the
/// top id — is a function of it, so prover and verifier builds must agree on
/// it: bump this with any change to the closed form, the fork constant, the cap
/// or the rule, as with a format change.
/// `the_partition_cost_model_is_pinned_to_its_version` pins its output.
///
/// - v1: every ECDAS, ECSM and KECCAK instance seeded on leaf 1, 2 and 3.
/// - v2: only `ECDAS[0]`, `ECSM[0]` and `KECCAK[0]` seeded there; their later
///   chunks fill by load, so no count of them overfills one leaf.
pub const PARTITION_COST_MODEL: u32 = 2;

/// The leaf that subtracts the COMMIT-bus target.
const CARRIER: usize = 0;

/// Goldilocks' two-adicity: the largest LDE domain is `2^32`.
const MAX_LOG2_LDE: u32 = 32;

/// What a consumer of a block proof receives besides the proof: the statement's
/// shape fields and each instance's trace length, in AIR order.
#[derive(Debug, Clone)]
pub struct BlockShape {
    pub table_counts: crate::TableCounts,
    pub runtime_page_ranges: Vec<crate::RuntimePageRange>,
    pub num_private_input_pages: usize,
    pub public_output_len: usize,
    pub trace_lengths: Vec<usize>,
}

impl BlockShape {
    /// The shape `proof` claims — what its prover hands a consumer with it.
    pub fn of_proof(proof: &crate::VmProof) -> Self {
        let view = stark::proof::view::MultiProofView::Owned(&proof.proof);
        Self {
            table_counts: proof.table_counts.clone(),
            runtime_page_ranges: proof.runtime_page_ranges.clone(),
            num_private_input_pages: proof.num_private_input_pages,
            public_output_len: proof.public_output.len(),
            trace_lengths: (0..view.len())
                .map(|i| view.get(i).trace_length())
                .collect(),
        }
    }
}

/// One instance of the block, as the plan derived it.
pub struct PlannedInstance {
    /// The AIR's name (`CPU[3]`, `PAGE:0x1000` …).
    pub name: String,
    pub challenge: TableChallengeShape,
    pub verify: TableVerifyShape,
    pub analysis: Analysis,
    /// The preprocessed root the host verifier takes from the AIR at this
    /// instance's leaf layout, when the AIR is preprocessed.
    pub precomputed_root: Option<Commitment>,
}

/// The plan's inputs that are a function of the ELF and the options alone —
/// DECODE's and the ELF data pages' preprocessed roots, recomputed on the host
/// from the ELF (never a prover's or a device's copy, and never the
/// process-wide record of one) — so a caller can compute them before the
/// block's shape exists. Constructed only by [`Self::compute`] (or
/// [`Self::compute_parallel`]), and tied to the ELF digest and the options it
/// was computed under.
#[derive(Debug, PartialEq, Eq)]
pub struct ElfConstants {
    elf_digest: [u8; 32],
    /// The options, as their `Debug` rendering: every field the roots depend on.
    opts: String,
    decode: Commitment,
    /// `(page base, row-pair root)` of each ELF data page.
    pages: Vec<(u64, Commitment)>,
}

impl ElfConstants {
    /// Compute the constants for `elf_bytes` under `opts`.
    pub fn compute(elf_bytes: &[u8], opts: &crate::ProofOptions) -> Result<Self, String> {
        Self::compute_with(elf_bytes, opts, false)
    }

    /// [`Self::compute`] with DECODE's root and every data page's computed at
    /// once on the caller's pool, the pages kept in ELF order: the same
    /// constants. The block ELF's twelve 2^18-row pages are most of the work and
    /// [`Self::compute`] commits them one at a time, so a wider pool barely
    /// shortens it.
    pub fn compute_parallel(elf_bytes: &[u8], opts: &crate::ProofOptions) -> Result<Self, String> {
        Self::compute_with(elf_bytes, opts, true)
    }

    fn compute_with(
        elf_bytes: &[u8],
        opts: &crate::ProofOptions,
        parallel: bool,
    ) -> Result<Self, String> {
        let elf = executor::elf::Elf::load(elf_bytes).map_err(|e| format!("ELF: {e}"))?;
        let configs = crate::tables::trace_builder::Traces::page_configs_from_elf(&elf);
        let decode = || {
            crate::tables::decode::commitment_from_elf(&elf, opts)
                .map_err(|e| format!("DECODE commitment: {e:?}"))
        };
        #[cfg(feature = "parallel")]
        let (decode, pages) = if parallel {
            let (decode, pages) = rayon::join(decode, || data_page_roots(&configs, opts, true));
            (decode?, pages)
        } else {
            (decode()?, data_page_roots(&configs, opts, false))
        };
        #[cfg(not(feature = "parallel"))]
        let (decode, pages) = (decode()?, data_page_roots(&configs, opts, parallel));
        Ok(Self {
            elf_digest: crate::statement::elf_digest(elf_bytes),
            opts: format!("{opts:?}"),
            decode,
            pages,
        })
    }
}

/// `(page base, row-pair root)` of each ELF data page among `configs` (those
/// with data, not private input), in their order; `parallel` commits them at
/// once on the caller's pool.
fn data_page_roots(
    configs: &[crate::tables::page::PageConfig],
    opts: &crate::ProofOptions,
    parallel: bool,
) -> Vec<(u64, Commitment)> {
    let data: Vec<_> = configs
        .iter()
        .filter(|c| c.init_values.is_some() && !c.is_private_input)
        .collect();
    let page = |c: &&crate::tables::page::PageConfig| {
        let root = crate::tables::page::compute_precomputed_commitment_with(
            c,
            opts,
            stark::leaf_layout::LeafLayout::RowPair,
        );
        (c.page_base, root)
    };
    #[cfg(feature = "parallel")]
    if parallel {
        use rayon::prelude::*;
        return data.par_iter().map(page).collect();
    }
    let _ = parallel;
    data.iter().map(page).collect()
}

/// Every program of a block's tree, derived from the trusted ELF, the options
/// and the block's shape. See the module docs.
pub struct BlockTreePlan {
    statement: super::block_replay::BlockStatementShape,
    elf_digest: [u8; 32],
    pc_start: u64,
    /// The DECODE instance.
    decode: usize,
    /// `(page base, instance)` of each ELF data page, in AIR order.
    data_pages: Vec<(u64, usize)>,
    instances: Vec<PlannedInstance>,
    partition: BlockPartition,
    /// The AIR set the shapes came from — the host verifier's. Kept for a
    /// caller that verifies the base over the very set (the harness's harvest).
    #[cfg_attr(not(test), allow(dead_code))]
    airs: crate::VmAirs,
}

impl BlockTreePlan {
    /// Derive the plan, or refuse a shape the host verifier refuses.
    ///
    /// The AIR set is `verify_proof_parts`' (and the harness's harvest's): page
    /// configs from the ELF and the shape's ranges, `VmAirs::new` with HALT and
    /// the production BITWISE, DECODE's root from the ELF.
    pub fn derive(
        elf_bytes: &[u8],
        opts: &crate::ProofOptions,
        shape: &BlockShape,
    ) -> Result<Self, String> {
        Self::derive_with(
            elf_bytes,
            opts,
            shape,
            &ElfConstants::compute(elf_bytes, opts)?,
        )
    }

    /// [`Self::derive`] over ELF constants computed ahead of the shape (beside
    /// the base, once per ELF): refuses constants of another ELF or options.
    pub fn derive_with(
        elf_bytes: &[u8],
        opts: &crate::ProofOptions,
        shape: &BlockShape,
        consts: &ElfConstants,
    ) -> Result<Self, String> {
        if consts.elf_digest != crate::statement::elf_digest(elf_bytes)
            || consts.opts != format!("{opts:?}")
        {
            return Err("ELF constants of another ELF or other options".to_string());
        }
        let elf = executor::elf::Elf::load(elf_bytes).map_err(|e| format!("ELF: {e}"))?;
        let page_configs = check_shape(&elf, opts, shape)?;
        let n = shape.trace_lengths.len();

        let decode = consts.decode;
        let airs = crate::VmAirs::new(
            &elf,
            opts,
            false,
            &page_configs,
            &shape.table_counts,
            Some(decode),
            true,
            None,
            Some(&consts.pages),
            None,
        );
        let refs = airs.air_refs();
        if refs.len() != n {
            return Err(format!("{} AIRs for {n} instances", refs.len()));
        }
        if !refs.iter().any(|a| a.has_aux_trace()) {
            return Err("a block uses LogUp".to_string());
        }

        let derive_one = |idx: usize| -> Result<PlannedInstance, String> {
            let air = refs[idx];
            let rows = shape.trace_lengths[idx];
            let (verify, analysis) = TableVerifyShape::derive(air, rows)
                .map_err(|e| format!("instance {idx} ({}): {e}", air.name()))?;
            let precomputed_root = if air.is_preprocessed() {
                Some(
                    super::epoch_verify::layout_precomputed_commitment(air, rows).ok_or_else(
                        || format!("instance {idx} ({}): no preprocessed root", air.name()),
                    )?,
                )
            } else {
                None
            };
            Ok(PlannedInstance {
                name: air.name().to_string(),
                challenge: TableChallengeShape::derive(air, idx, n, rows),
                verify,
                analysis,
                precomputed_root,
            })
        };
        #[cfg(feature = "parallel")]
        let instances: Vec<PlannedInstance> = {
            use rayon::prelude::*;
            (0..n)
                .into_par_iter()
                .map(derive_one)
                .collect::<Result<_, _>>()?
        };
        #[cfg(not(feature = "parallel"))]
        let instances: Vec<PlannedInstance> = (0..n).map(derive_one).collect::<Result<_, _>>()?;

        // DECODE and the ELF data pages: the instances whose roots the
        // attestation id covers.
        let mut decode_idx = None;
        let mut data_pages = Vec::new();
        let mut page = 0usize;
        for (idx, inst) in instances.iter().enumerate() {
            if inst.name == "DECODE" {
                if decode_idx.replace(idx).is_some() {
                    return Err("two DECODE instances".to_string());
                }
            } else if inst.name.starts_with("PAGE:") {
                let config = page_configs
                    .get(page)
                    .ok_or("more PAGE instances than configs")?;
                page += 1;
                if config.init_values.is_some() && !config.is_private_input {
                    // The root is the host's recompute from the ELF, never the
                    // process-wide record a prover's device wrote: the row-pair
                    // root was supplied from `consts`; any other layout's is
                    // recomputed here and must agree with the AIR's.
                    let layout =
                        stark::leaf_layout::table_leaf_layout(refs[idx], shape.trace_lengths[idx]);
                    let host = match layout {
                        stark::leaf_layout::LeafLayout::RowPair => consts
                            .pages
                            .iter()
                            .find(|(base, _)| *base == config.page_base)
                            .map(|(_, root)| *root),
                        other => Some(crate::tables::page::compute_precomputed_commitment_with(
                            config, opts, other,
                        )),
                    };
                    if host.is_none() || inst.precomputed_root != host {
                        return Err(format!(
                            "data page {idx} at 0x{:x}: its root is not the host's recompute",
                            config.page_base
                        ));
                    }
                    data_pages.push((config.page_base, idx));
                }
            }
        }
        if page != page_configs.len() {
            return Err("one PAGE instance per page config".to_string());
        }
        let decode = decode_idx.ok_or("a block has a DECODE instance")?;
        if instances[decode].precomputed_root.is_none() {
            return Err("DECODE is preprocessed".to_string());
        }

        let costs: Vec<usize> = instances.iter().map(|i| instance_cost(&i.verify)).collect();
        let names: Vec<&str> = instances.iter().map(|i| i.name.as_str()).collect();
        let partition = partition_for(&names, &costs)?;

        Ok(Self {
            statement: super::block_replay::BlockStatementShape {
                public_output_len: shape.public_output_len,
                table_counts: crate::statement::table_count_values(&shape.table_counts),
                num_private_input_pages: shape.num_private_input_pages as u64,
                fri_final_poly_log_degree: opts.fri_final_poly_log_degree,
                page_ranges: shape
                    .runtime_page_ranges
                    .iter()
                    .map(|r| (r.base, r.count))
                    .collect(),
            },
            elf_digest: crate::statement::elf_digest(elf_bytes),
            pc_start: elf.entry_point,
            decode,
            data_pages,
            instances,
            partition,
            airs,
        })
    }

    pub fn statement(&self) -> &super::block_replay::BlockStatementShape {
        &self.statement
    }

    pub fn elf_digest(&self) -> &[u8; 32] {
        &self.elf_digest
    }

    pub fn num_instances(&self) -> usize {
        self.instances.len()
    }

    pub fn instance(&self, i: usize) -> &PlannedInstance {
        &self.instances[i]
    }

    pub fn instances(&self) -> &[PlannedInstance] {
        &self.instances
    }

    /// The AIR set the plan derived its shapes from — the host verifier's.
    pub(crate) fn airs(&self) -> &crate::VmAirs {
        &self.airs
    }

    /// The ELF-derived inputs the leaves' attestation id covers, assembled from
    /// the instances' own preprocessed roots — the roots every leaf absorbs.
    pub fn attested(&self) -> AttestedInputs {
        let root = |i: usize| {
            self.instances[i]
                .precomputed_root
                .expect("checked at derive")
        };
        AttestedInputs {
            elf_digest: self.elf_digest,
            pc_start: self.pc_start,
            decode: root(self.decode),
            pages: self
                .data_pages
                .iter()
                .map(|&(base, i)| (base, root(i)))
                .collect(),
        }
    }

    pub fn partition(&self) -> &BlockPartition {
        &self.partition
    }

    /// The partition's cost-model version ([`PARTITION_COST_MODEL`]).
    pub fn cost_model(&self) -> u32 {
        PARTITION_COST_MODEL
    }

    /// The one leaf that subtracts the COMMIT-bus target.
    pub fn carrier(&self) -> usize {
        CARRIER
    }

    /// Per instance, its in-guest verification cost: the legs' closed form plus
    /// the fork's transcript and grinding.
    pub fn costs(&self) -> Vec<usize> {
        self.instances
            .iter()
            .map(|i| instance_cost(&i.verify))
            .collect()
    }

    /// The layout a leaf and a non-top node publish.
    pub fn child_layout(&self) -> BlockLayout {
        BlockLayout::child(
            super::proof_arena::words_per_root(),
            self.statement.out_halves(),
        )
    }

    /// The layout the top node publishes.
    pub fn top_layout(&self) -> BlockLayout {
        BlockLayout::top(
            super::proof_arena::words_per_root(),
            self.statement.out_halves(),
        )
    }

    /// Leaf `k`'s program, emitted and validated.
    pub fn leaf_program(&self, k: usize) -> Result<LfmProgram, String> {
        if k >= self.partition.num_leaves() {
            return Err(format!("no leaf {k}"));
        }
        let mut b = builder();
        super::block_leaf::emit_block_leaf(&mut b, self, k);
        finish(b)
    }

    /// The node over `children`, the top when `top`, emitted and validated.
    pub fn node_program(&self, children: &[DerivedChild], top: bool) -> Result<LfmProgram, String> {
        let shapes: Vec<_> = children.iter().map(DerivedChild::shape).collect();
        let mut b = builder();
        super::block_node::emit_block_node(
            &mut b,
            &BlockNodeInputs {
                children: &shapes,
                layout: self.child_layout(),
                top,
            },
        );
        finish(b)
    }

    /// The levels above the leaves: fan-in [`block_fan_in`], the last node of a
    /// level taking what is left, the last level's single node the top. A
    /// one-leaf block still gets a top node: the bus closes there.
    pub fn levels(&self) -> Vec<super::per_table_aggregator::Level> {
        use super::per_table_aggregator::{Level, tree_shape};
        let mut shape = tree_shape(self.partition.num_leaves(), block_fan_in());
        if shape.is_empty() {
            shape.push(Level { arities: vec![1] });
        }
        shape
    }

    /// The top program's artifacts, with no proof: every leaf emitted, its
    /// artifacts built, each node emitted over its children's derived shapes —
    /// level by level, under the tree's options `wrap_opts`.
    pub fn derive_top(&self, wrap_opts: &crate::ProofOptions) -> Result<LfmArtifacts, String> {
        self.derive_top_timed(wrap_opts).map(|(top, _)| top)
    }

    /// [`Self::derive_top`] and its stopwatch, one [`PhaseTimes`] per level (the
    /// leaves, then each node level). A level's programs are emitted and built
    /// in parallel, their device commits under the derive gate's running total
    /// ([`super::derive_gate`]), which this arms: the block verifier's
    /// derivation, with no prove beside it.
    pub(crate) fn derive_top_timed(
        &self,
        wrap_opts: &crate::ProofOptions,
    ) -> Result<(LfmArtifacts, Vec<PhaseTimes>), String> {
        let _gate = super::derive_gate::arm();
        let (mut tree, phases) = self.derive_levels(
            wrap_opts,
            &|program| artifacts_of(program, wrap_opts),
            false,
        )?;
        match tree.pop().map(<[_; 1]>::try_from) {
            Some(Ok([(_, artifacts)])) => Ok((artifacts, phases)),
            _ => Err("the tree does not close to one node".to_string()),
        }
    }

    /// Every program of the tree with its artifacts, level by level (the leaves
    /// in leaf order first, the top last), built by `build` — what a prover
    /// proves from when it derives the tree ahead of the base's proof. The
    /// programs are [`Self::derive_top`]'s.
    #[cfg_attr(not(test), allow(dead_code))]
    pub(crate) fn derive_tree(
        &self,
        wrap_opts: &crate::ProofOptions,
        build: &(dyn Fn(&LfmProgram) -> LfmArtifacts + Sync),
    ) -> Result<(TreePrograms, Vec<PhaseTimes>), String> {
        self.derive_levels(wrap_opts, build, true)
    }

    /// The tree level by level; below the top, each level's programs and
    /// artifacts are kept only when `keep`, and held until the level is done
    /// only under [`derive_holds_levels`].
    fn derive_levels(
        &self,
        wrap_opts: &crate::ProofOptions,
        build: &(dyn Fn(&LfmProgram) -> LfmArtifacts + Sync),
        keep: bool,
    ) -> Result<(TreePrograms, Vec<PhaseTimes>), String> {
        let words = self.child_layout().total();
        let mut phases = Vec::new();
        let mut kept = Vec::new();
        let hold = keep || derive_holds_levels();
        let leaves: Vec<usize> = (0..self.partition.num_leaves()).collect();
        let mut level = derive_level(
            &leaves,
            |&k| self.leaf_program(k),
            build,
            wrap_opts,
            words,
            hold,
            &mut phases,
        )?;
        let levels = self.levels();
        for (lv, arities) in levels.iter().enumerate() {
            let top = lv + 1 == levels.len();
            let mut groups = Vec::with_capacity(arities.arities.len());
            let mut done = Vec::new();
            let mut rest = level.into_iter();
            for &a in &arities.arities {
                let mut kids: Vec<DerivedChild> = Vec::with_capacity(a);
                for (held, child) in rest.by_ref().take(a) {
                    kids.push(child);
                    if keep && let Some(pair) = held {
                        done.push(pair);
                    }
                }
                if kids.len() != a {
                    return Err(format!("level {}: arities overrun the children", lv + 1));
                }
                groups.push(kids);
            }
            if rest.next().is_some() {
                return Err(format!("level {}: arities leave children over", lv + 1));
            }
            if keep {
                kept.push(done);
            }
            // The top's program and artifacts are the derivation's result.
            level = derive_level(
                &groups,
                |kids| self.node_program(kids, top),
                build,
                wrap_opts,
                words,
                hold || top,
                &mut phases,
            )?;
        }
        if level.len() != 1 {
            return Err(format!("the tree closes to {} nodes", level.len()));
        }
        let top = level
            .into_iter()
            .map(|(held, _)| held.ok_or("the top level keeps its program"))
            .collect::<Result<Vec<_>, _>>()?;
        kept.push(top);
        Ok((kept, phases))
    }

    /// One instance's shapes, mutable — a mutation, test-only: a leaf emitted
    /// from any other shape is another program.
    #[cfg(test)]
    pub(crate) fn instance_mut(&mut self, i: usize) -> &mut PlannedInstance {
        &mut self.instances[i]
    }

    /// Replace the partition — a different tree, test-only: what a prover may
    /// emit leaves over, refused at the final check.
    #[cfg(test)]
    pub(crate) fn with_partition(mut self, partition: BlockPartition) -> Self {
        assert_eq!(partition.num_instances(), self.instances.len());
        self.partition = partition;
        self
    }
}

/// The block tree's fan-in: children per interior node. Four: the record
/// block's 8 leaves prove 3 nodes instead of 7 (FAST 399: recursion −1.02 s
/// against two; FAST 450: three is +1.15 s against four). The epoch tree's
/// [`super::per_table_aggregator::FAN_IN`] is a separate constant.
pub const BLOCK_FAN_IN: usize = 4;

/// The fan-in the plan derives under: [`BLOCK_FAN_IN`], or
/// `NOEPOCH_BLOCK_FAN_IN` (2 to 8) for a measurement. It is a tree-format
/// parameter: every node program, and so the top, depends on it. A verifier
/// configured otherwise than the prover derives another top and refuses
/// (completeness, never soundness), as with the presets' `ZfFormat`.
pub fn block_fan_in() -> usize {
    static FAN_IN: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *FAN_IN.get_or_init(|| match std::env::var("NOEPOCH_BLOCK_FAN_IN") {
        Err(_) => BLOCK_FAN_IN,
        Ok(v) => match v.parse::<usize>() {
            Ok(n) if (2..=8).contains(&n) => n,
            _ => panic!("NOEPOCH_BLOCK_FAN_IN must be 2 to 8, got `{v}`"),
        },
    })
}

/// `verify_proof_parts`' pre-checks on a claimed shape, before any AIR is
/// built: the accelerator shape (`BlockChunked`, the block verifier's), the
/// private-input-page bound, the page configs (ELF pages plus the runtime
/// ranges: aligned, non-empty, disjoint from the ELF's), the instance count
/// against the trace lengths, the chunked tables' heights (KECCAK, KECCAK_RND,
/// ECSM and ECDAS under their caps) and each trace length (a power of two
/// inside the field's two-adicity). Returns the page configs.
pub fn check_shape(
    elf: &executor::elf::Elf,
    opts: &crate::ProofOptions,
    shape: &BlockShape,
) -> Result<Vec<crate::tables::page::PageConfig>, String> {
    let n = shape.trace_lengths.len();
    shape
        .table_counts
        .validate_for(crate::AcceleratorShape::BlockChunked)
        .map_err(|e| format!("table counts: {e:?}"))?;
    let max_pages = crate::tables::page::max_private_input_pages();
    if shape.num_private_input_pages > max_pages {
        return Err(format!(
            "{} private-input pages, over the bound {max_pages}",
            shape.num_private_input_pages
        ));
    }
    let page_configs = crate::tables::trace_builder::Traces::page_configs_from_elf_and_runtime(
        elf,
        &shape.runtime_page_ranges,
        shape.num_private_input_pages,
        n,
    )
    .map_err(|e| format!("page configs: {e:?}"))?;
    let expected = shape
        .table_counts
        .total()
        .and_then(|t| t.checked_add(crate::FIXED_TABLE_COUNT))
        .and_then(|t| t.checked_add(page_configs.len()))
        .ok_or("the declared table counts overflow")?;
    if expected != n {
        return Err(format!("{expected} instances declared, {n} trace lengths"));
    }
    shape
        .table_counts
        .check_heights_for(crate::AcceleratorShape::BlockChunked, |i| {
            shape.trace_lengths[i]
        })
        .map_err(|e| format!("table heights: {e:?}"))?;
    let log2_blowup = (opts.blowup_factor as usize).trailing_zeros();
    for (i, &rows) in shape.trace_lengths.iter().enumerate() {
        if !rows.is_power_of_two() || rows.trailing_zeros() + log2_blowup > MAX_LOG2_LDE {
            return Err(format!("instance {i}: no trace of {rows} rows"));
        }
    }
    Ok(page_configs)
}

/// One instance's in-guest cost: the legs' closed form plus the fork.
fn instance_cost(verify: &TableVerifyShape) -> usize {
    super::epoch_verify::table_permutations_for(verify, super::edsl::WrapHash::production())
        + FORK_PERMS
}

/// D-NOEPOCH §12.2's partition: the rule over `⌈Σ cost / cap⌉` leaves, one more
/// leaf while the heaviest leaf is over the cap and another leaf can help (a
/// block whose instances are all at or under the cap). Pure in the shape.
fn partition_for(names: &[&str], costs: &[usize]) -> Result<BlockPartition, String> {
    let total: usize = costs.iter().sum();
    let widest = costs.iter().copied().max().unwrap_or(0);
    let mut k = total.div_ceil(LEAF_PERMS_CAP).clamp(1, names.len().max(1));
    loop {
        let p = partition_by_rule(names, costs, k)?;
        let heaviest = p
            .leaves()
            .iter()
            .map(|l| l.iter().map(|&i| costs[i]).sum::<usize>())
            .max()
            .unwrap_or(0);
        if heaviest <= LEAF_PERMS_CAP || widest > LEAF_PERMS_CAP || k == names.len() {
            return Ok(p);
        }
        k += 1;
    }
}

fn builder() -> LfmBuilder {
    LfmBuilder::new().with_wrap_hash(super::edsl::WrapHash::production())
}

fn finish(b: LfmBuilder) -> Result<LfmProgram, String> {
    let program = compile(b.finish());
    super::validator::validate(&program).map_err(|e| format!("not admissible: {e:?}"))?;
    Ok(program)
}

/// A tree program's artifacts, under the block hasher.
pub fn artifacts_of(program: &LfmProgram, wrap_opts: &crate::ProofOptions) -> LfmArtifacts {
    super::program_census::build_artifacts_counted(
        program,
        wrap_opts,
        crate::hash_pin::BLOCK_HASHER,
    )
}

/// A tree's programs with their artifacts, level by level: the leaves first, in
/// leaf order, then each node level, the top last.
pub(crate) type TreePrograms = Vec<Vec<(LfmProgram, LfmArtifacts)>>;

/// `LAMBDA_VM_BLOCK_DERIVE_BUILDS=<n>`: a level's items are emitted and built
/// in windows of `n` while a tree is derived ([`BlockTreePlan::derive_top`],
/// so the block verifier, and [`BlockTreePlan::derive_tree`]), each window in
/// parallel; unset, the whole level is one window. Nothing is held while a
/// window runs: a place taken in a rayon job and held across the build, which
/// waits inside rayon, can be asked for again by a job its thread steals there
/// (#1014's W3 hang, BIG 569). The card's bound is the derive gate's running
/// total in bytes ([`super::derive_gate`]), not this count: the median block's
/// unbounded verifier peaked at 31.85 GiB of a 32 GiB card (BIG 480), the p90
/// block's ran out (BIG 123). Scheduling only: every artifact is a pure
/// function of its program and the options, so the derived tree and the top
/// program do not depend on it.
fn derive_builds_bound() -> Option<usize> {
    static BOUND: std::sync::OnceLock<Option<usize>> = std::sync::OnceLock::new();
    *BOUND.get_or_init(|| match std::env::var("LAMBDA_VM_BLOCK_DERIVE_BUILDS") {
        Ok(v) if !v.is_empty() => Some(v.parse::<usize>().ok().filter(|&n| n >= 1).unwrap_or_else(
            || panic!("LAMBDA_VM_BLOCK_DERIVE_BUILDS must be a positive integer, got `{v}`"),
        )),
        _ => None,
    })
}

/// `f` over `items` in order, in windows of `window` (all of them at once when
/// `None`), each window in parallel where there is a pool; the first error
/// stops it.
fn map_in_windows<T: Sync, R: Send>(
    items: &[T],
    window: Option<usize>,
    f: impl Fn(&T) -> Result<R, String> + Sync + Send,
) -> Result<Vec<R>, String> {
    let mut done = Vec::with_capacity(items.len());
    for part in items.chunks(window.unwrap_or(items.len()).max(1)) {
        #[cfg(feature = "parallel")]
        {
            use rayon::prelude::*;
            done.extend(
                part.par_iter()
                    .map(&f)
                    .collect::<Result<Vec<_>, String>>()?,
            );
        }
        #[cfg(not(feature = "parallel"))]
        done.extend(part.iter().map(&f).collect::<Result<Vec<_>, String>>()?);
    }
    Ok(done)
}

/// `LAMBDA_VM_BLOCK_DERIVE_HOLD=level`: [`BlockTreePlan::derive_top`] holds each
/// level's programs and artifacts until the whole level is derived, as it did
/// before (the A/B control). Unset, each item's program and artifacts are
/// dropped as soon as its child's shape is derived, so a level holds only the
/// items in flight rather than all of them: the median block's leaf level is 56
/// programs of ≈ 0.42 GiB held each. Scheduling only: the derived shapes, and so
/// the top program, are the same either way.
fn derive_holds_levels() -> bool {
    static HOLD: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *HOLD.get_or_init(
        || match std::env::var("LAMBDA_VM_BLOCK_DERIVE_HOLD").ok().as_deref() {
            None | Some("") => false,
            Some("level") => true,
            Some(v) => panic!("LAMBDA_VM_BLOCK_DERIVE_HOLD must be `level`, got `{v}`"),
        },
    )
}

/// One item of a derived level: its program and artifacts when the level keeps
/// them, and its shape as a child.
type DerivedItem = (Option<(LfmProgram, LfmArtifacts)>, DerivedChild);

/// One level of a tree's derivation: each item's program emitted, its artifacts
/// built by `build` and its shape as a child derived, in parallel and in order,
/// in windows of [`derive_builds_bound`]. Without `keep`, each item's
/// program and artifacts are dropped as soon as its child is derived.
/// Pushes the level's stopwatch onto `phases`.
fn derive_level<T: Sync>(
    items: &[T],
    emit: impl Fn(&T) -> Result<LfmProgram, String> + Sync,
    build: &(dyn Fn(&LfmProgram) -> LfmArtifacts + Sync),
    wrap_opts: &crate::ProofOptions,
    words: usize,
    keep: bool,
    phases: &mut Vec<PhaseTimes>,
) -> Result<Vec<DerivedItem>, String> {
    let t = Instant::now();
    type Derived = (DerivedItem, f64, f64);
    let one = |item: &T| -> Result<Derived, String> {
        let t = Instant::now();
        let program = emit(item)?;
        let emitted = t.elapsed().as_secs_f64();
        let t = Instant::now();
        let artifacts = build(&program);
        let derived = DerivedChild::from_artifacts(&artifacts, wrap_opts, words)?;
        let built = t.elapsed().as_secs_f64();
        Ok((
            (keep.then_some((program, artifacts)), derived),
            emitted,
            built,
        ))
    };
    let done = map_in_windows(items, derive_builds_bound(), one)?;
    let mut phase = PhaseTimes {
        programs: done.len(),
        ..PhaseTimes::default()
    };
    let level = done
        .into_iter()
        .map(|(item, emitted, built)| {
            phase.emit += emitted;
            phase.build += built;
            item
        })
        .collect();
    phase.wall = t.elapsed().as_secs_f64();
    phases.push(phase);
    Ok(level)
}

/// One level of a tree's derivation, in seconds: its wall time, and its
/// programs' emit and build summed (they run in parallel).
#[derive(Clone, Debug, Default)]
pub(crate) struct PhaseTimes {
    pub programs: usize,
    pub wall: f64,
    pub emit: f64,
    pub build: f64,
}

/// The block verifier's stopwatch, in seconds.
#[derive(Clone, Debug, Default)]
pub(crate) struct VerifyTimes {
    /// The ELF constants (zero when the caller supplied them).
    pub constants: f64,
    /// The plan, past its constants.
    pub plan: f64,
    /// The top program's derivation, level by level (the leaves first).
    pub levels: Vec<PhaseTimes>,
    /// The top proof verified against the derived program, and its words
    /// checked against the block.
    pub check: f64,
}

impl std::fmt::Display for VerifyTimes {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "constants {:.2} · plan {:.2}", self.constants, self.plan)?;
        for (i, l) in self.levels.iter().enumerate() {
            let name = if i == 0 {
                "leaves".to_string()
            } else {
                format!("L{i}")
            };
            write!(
                f,
                " · {name} {} in {:.2} (emit Σ {:.2}, build Σ {:.2})",
                l.programs, l.wall, l.emit, l.build
            )?;
        }
        write!(f, " · check {:.2}", self.check)
    }
}

/// The public output as the statement's 32-bit halves, little-endian, the last
/// one zero-padded.
pub fn le_halves(bytes: &[u8]) -> Vec<FE> {
    bytes
        .chunks(4)
        .map(|c| {
            let mut le = [0u8; 4];
            le[..c.len()].copy_from_slice(c);
            FE::from(u64::from(u32::from_le_bytes(le)))
        })
        .collect()
}

/// Whether `words`, a top proof's published words, claim the block: the trusted
/// ELF's attestation id and `public_output`, in the top layout. The state words
/// are free here (the nodes bind them across children).
pub fn top_claims(plan: &BlockTreePlan, words: &[(u32, LfmWord)], public_output: &[u8]) -> bool {
    let layout = plan.top_layout();
    if words.len() != layout.total()
        || words
            .iter()
            .enumerate()
            .any(|(i, (at, _))| *at as usize != i)
    {
        return false;
    }
    let id = super::programs::program_id_words(&plan.attested().program_id());
    let halves = le_halves(public_output);
    plan.statement.public_output_len == public_output.len()
        && (0..2).all(|h| words[layout.id(h)].1 == id[h])
        && halves
            .iter()
            .enumerate()
            .all(|(i, v)| words[layout.out_half(i)].1 == base_word(*v))
}

/// ★ The no-epoch block's verifier, over its tree's top proof: derive the plan
/// ([`BlockTreePlan::derive`]) and the top program
/// ([`BlockTreePlan::derive_top`]) under the block presets — the base
/// under [`super::proof::block_base_options`], the tree under
/// [`super::proof::aggregation_wrap_options`], verifier constants, never a
/// caller's or a prover's — verify `top` against that program, and check that it
/// publishes the ELF's attestation id and `public_output`. Returns the id of the
/// top program the proof verified against.
///
/// Both presets stamp the process's `ZfFormat`, which environment knobs set, so
/// the verifier's environment is part of the derived identity: a verifier
/// configured otherwise than the prover derives another top and refuses
/// (completeness, never soundness).
pub fn verify_block_tree(
    elf_bytes: &[u8],
    shape: &BlockShape,
    public_output: &[u8],
    top: &super::proof::LfmProof,
) -> Result<Commitment, String> {
    verify_block_tree_timed(elf_bytes, None, shape, public_output, top).map(|(id, _)| id)
}

/// [`verify_block_tree`] over ELF constants computed ahead
/// ([`ElfConstants::compute`] under [`super::proof::block_base_options`]), so a
/// consumer verifying many blocks of one ELF computes them once. Refuses
/// constants of another ELF or other options, as [`BlockTreePlan::derive_with`]
/// does.
pub fn verify_block_tree_with(
    elf_bytes: &[u8],
    consts: &ElfConstants,
    shape: &BlockShape,
    public_output: &[u8],
    top: &super::proof::LfmProof,
) -> Result<Commitment, String> {
    verify_block_tree_timed(elf_bytes, Some(consts), shape, public_output, top).map(|(id, _)| id)
}

/// [`verify_block_tree`], over `consts` when given, and its stopwatch.
pub(crate) fn verify_block_tree_timed(
    elf_bytes: &[u8],
    consts: Option<&ElfConstants>,
    shape: &BlockShape,
    public_output: &[u8],
    top: &super::proof::LfmProof,
) -> Result<(Commitment, VerifyTimes), String> {
    verify_under(
        elf_bytes,
        &super::proof::block_base_options(),
        &super::proof::aggregation_wrap_options(),
        consts,
        shape,
        public_output,
        top,
    )
}

/// [`verify_block_tree`] under other presets — the fixture blocks', whose base
/// runs smaller options.
#[cfg(test)]
pub(crate) fn verify_block_tree_under(
    elf_bytes: &[u8],
    opts: &crate::ProofOptions,
    wrap_opts: &crate::ProofOptions,
    consts: Option<&ElfConstants>,
    shape: &BlockShape,
    public_output: &[u8],
    top: &super::proof::LfmProof,
) -> Result<Commitment, String> {
    verify_under(
        elf_bytes,
        opts,
        wrap_opts,
        consts,
        shape,
        public_output,
        top,
    )
    .map(|(id, _)| id)
}

fn verify_under(
    elf_bytes: &[u8],
    opts: &crate::ProofOptions,
    wrap_opts: &crate::ProofOptions,
    consts: Option<&ElfConstants>,
    shape: &BlockShape,
    public_output: &[u8],
    top: &super::proof::LfmProof,
) -> Result<(Commitment, VerifyTimes), String> {
    let mut times = VerifyTimes::default();
    let computed;
    let consts = match consts {
        Some(consts) => consts,
        None => {
            let t = Instant::now();
            computed = ElfConstants::compute(elf_bytes, opts)?;
            times.constants = t.elapsed().as_secs_f64();
            &computed
        }
    };
    let t = Instant::now();
    let plan = BlockTreePlan::derive_with(elf_bytes, opts, shape, consts)?;
    times.plan = t.elapsed().as_secs_f64();
    let (artifacts, levels) = plan.derive_top_timed(wrap_opts)?;
    times.levels = levels;
    let t = Instant::now();
    if !super::proof::verify_against_artifacts(&artifacts, &top.proof, &top.public_words, wrap_opts)
    {
        return Err("the top proof does not verify against the derived top program".to_string());
    }
    if !top_claims(&plan, &top.public_words, public_output) {
        return Err("the top proof does not claim this ELF's id and this output".to_string());
    }
    times.check = t.elapsed().as_secs_f64();
    Ok((artifacts.program_id, times))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    /// The count cap's windows: at most `n` items at once, every item once,
    /// in order, and the first error stops the level.
    #[test]
    fn a_level_in_windows_holds_its_bound_and_its_order() {
        let items: Vec<usize> = (0..23).collect();
        for window in [None, Some(1), Some(2), Some(5), Some(64)] {
            let inside = AtomicUsize::new(0);
            let most = AtomicUsize::new(0);
            let done = map_in_windows(&items, window, |&i| {
                let now = inside.fetch_add(1, Ordering::SeqCst) + 1;
                most.fetch_max(now, Ordering::SeqCst);
                std::thread::sleep(std::time::Duration::from_millis(2));
                inside.fetch_sub(1, Ordering::SeqCst);
                Ok(i * 3)
            })
            .expect("no item fails");
            assert_eq!(done, items.iter().map(|i| i * 3).collect::<Vec<_>>());
            if let Some(n) = window {
                assert!(most.load(Ordering::SeqCst) <= n, "window {n}");
            }
        }
        let failed = map_in_windows(&items, Some(4), |&i| {
            if i == 9 {
                Err(format!("item {i}"))
            } else {
                Ok(i)
            }
        });
        assert_eq!(failed, Err("item 9".to_string()));
    }

    /// The data pages' roots committed at once are the ones committed a page at
    /// a time, in the same order, at any pool width; a zero page or a private
    /// input page has no root here.
    #[test]
    fn the_data_page_roots_at_once_are_the_page_at_a_time_ones() {
        use crate::tables::page::PageConfig;
        let opts = crate::recursion::Preset::Blowup4.options();
        let bytes = |seed: u8, n: usize| -> Vec<u8> {
            (0..n)
                .map(|i| (i as u8).wrapping_mul(31).wrapping_add(seed))
                .collect()
        };
        let configs = vec![
            PageConfig::with_data(0x0, bytes(1, 4096)),
            PageConfig::zero_init(0x40000),
            PageConfig::with_data(0x80000, bytes(2, 300)),
            PageConfig::with_private_input(0xff000000, bytes(4, 64)),
            PageConfig::with_data(0xc0000, bytes(3, 70_000)),
        ];
        let serial = data_page_roots(&configs, &opts, false);
        assert_eq!(
            serial.iter().map(|(base, _)| *base).collect::<Vec<_>>(),
            vec![0x0, 0x80000, 0xc0000],
            "the data pages, in order"
        );
        assert!(
            serial[0].1 != serial[1].1 && serial[1].1 != serial[2].1,
            "each page its own root"
        );
        for threads in [1, 3, 8] {
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .expect("a pool");
            assert_eq!(
                pool.install(|| data_page_roots(&configs, &opts, true)),
                serial,
                "{threads} thread(s)"
            );
        }
    }
}
