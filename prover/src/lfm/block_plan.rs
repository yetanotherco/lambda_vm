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
        let elf = executor::elf::Elf::load(elf_bytes).map_err(|e| format!("ELF: {e}"))?;
        let page_configs = check_shape(&elf, opts, shape)?;
        let n = shape.trace_lengths.len();

        let decode = crate::tables::decode::commitment_from_elf(&elf, opts)
            .map_err(|e| format!("DECODE commitment: {e:?}"))?;
        let airs = crate::VmAirs::new(
            &elf,
            opts,
            false,
            &page_configs,
            &shape.table_counts,
            Some(decode),
            true,
            None,
            None,
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
                    if inst.precomputed_root.is_none() {
                        return Err(format!("data page {idx} is not preprocessed"));
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
    #[cfg(test)]
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

    /// The levels above the leaves: fan-in
    /// [`super::per_table_aggregator::FAN_IN`], leftovers in arity-1 nodes, the
    /// last level's single node the top. A one-leaf block still gets a top
    /// node: the bus closes there.
    pub fn levels(&self) -> Vec<super::per_table_aggregator::Level> {
        use super::per_table_aggregator::{FAN_IN, Level, tree_shape};
        let mut shape = tree_shape(self.partition.num_leaves(), FAN_IN);
        if shape.is_empty() {
            shape.push(Level { arities: vec![1] });
        }
        shape
    }

    /// The top program's artifacts, with no proof: every leaf emitted, its
    /// artifacts built, each node emitted over its children's derived shapes —
    /// level by level, under the tree's options `wrap_opts`.
    pub fn derive_top(&self, wrap_opts: &crate::ProofOptions) -> Result<LfmArtifacts, String> {
        let words = self.child_layout().total();
        let child = |program: &LfmProgram| -> Result<(LfmArtifacts, DerivedChild), String> {
            let artifacts = artifacts_of(program, wrap_opts);
            let derived = DerivedChild::from_artifacts(&artifacts, wrap_opts, words)?;
            Ok((artifacts, derived))
        };
        let leaf = |k: usize| child(&self.leaf_program(k)?);
        #[cfg(feature = "parallel")]
        let mut level: Vec<(LfmArtifacts, DerivedChild)> = {
            use rayon::prelude::*;
            (0..self.partition.num_leaves())
                .into_par_iter()
                .map(leaf)
                .collect::<Result<_, String>>()?
        };
        #[cfg(not(feature = "parallel"))]
        let mut level: Vec<(LfmArtifacts, DerivedChild)> = (0..self.partition.num_leaves())
            .map(leaf)
            .collect::<Result<_, String>>()?;
        let levels = self.levels();
        for (lv, arities) in levels.iter().enumerate() {
            let top = lv + 1 == levels.len();
            let mut next = Vec::with_capacity(arities.arities.len());
            let mut rest = level.into_iter();
            for &a in &arities.arities {
                let kids: Vec<DerivedChild> = rest.by_ref().take(a).map(|(_, d)| d).collect();
                if kids.len() != a {
                    return Err(format!("level {}: arities overrun the children", lv + 1));
                }
                next.push(child(&self.node_program(&kids, top)?)?);
            }
            if rest.next().is_some() {
                return Err(format!("level {}: arities leave children over", lv + 1));
            }
            level = next;
        }
        match <[_; 1]>::try_from(level) {
            Ok([(artifacts, _)]) => Ok(artifacts),
            Err(level) => Err(format!("the tree closes to {} nodes", level.len())),
        }
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

/// `verify_proof_parts`' pre-checks on a claimed shape, before any AIR is
/// built: the accelerator shape (`KeccakRndChunked`, the block verifier's), the
/// private-input-page bound, the page configs (ELF pages plus the runtime
/// ranges: aligned, non-empty, disjoint from the ELF's), the instance count
/// against the trace lengths, and each trace length (a power of two inside the
/// field's two-adicity). Returns the page configs.
pub fn check_shape(
    elf: &executor::elf::Elf,
    opts: &crate::ProofOptions,
    shape: &BlockShape,
) -> Result<Vec<crate::tables::page::PageConfig>, String> {
    let n = shape.trace_lengths.len();
    shape
        .table_counts
        .validate_for(crate::AcceleratorShape::KeccakRndChunked)
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

/// The plan and its top program's artifacts, from the trusted ELF, the base
/// options, the tree's options and the claimed shape — no proof read.
pub fn derive_block_top(
    elf_bytes: &[u8],
    opts: &crate::ProofOptions,
    wrap_opts: &crate::ProofOptions,
    shape: &BlockShape,
) -> Result<(BlockTreePlan, LfmArtifacts), String> {
    let plan = BlockTreePlan::derive(elf_bytes, opts, shape)?;
    let top = plan.derive_top(wrap_opts)?;
    Ok((plan, top))
}

/// ★ The no-epoch block's verifier, over its tree's top proof: derive the plan
/// and the top program ([`derive_block_top`]); verify `top` against that
/// program; check that it publishes the ELF's attestation id and
/// `public_output`.
pub fn verify_block_tree(
    elf_bytes: &[u8],
    opts: &crate::ProofOptions,
    wrap_opts: &crate::ProofOptions,
    shape: &BlockShape,
    public_output: &[u8],
    top: &super::proof::LfmProof,
) -> Result<(), String> {
    let (plan, artifacts) = derive_block_top(elf_bytes, opts, wrap_opts, shape)?;
    if !super::proof::verify_against_artifacts(&artifacts, &top.proof, &top.public_words, wrap_opts)
    {
        return Err("the top proof does not verify against the derived top program".to_string());
    }
    if !top_claims(&plan, &top.public_words, public_output) {
        return Err("the top proof does not claim this ELF's id and this output".to_string());
    }
    Ok(())
}
