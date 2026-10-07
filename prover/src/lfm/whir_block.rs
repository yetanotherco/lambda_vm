//! The no-epoch WHIR BLOCK's recursion: the tree plan, the leaf program and its
//! arena (D-NOEPOCH §12.5, I-NOEPOCH-W §9).
//!
//! A block proof ([`crate::block_whir`]) is ONE multilinear proof: the
//! statement and the partition, every group's roots and every prepared table's
//! derived roots, then `(z, α, β)`, then each group on its own fork `S_post ‖ g`
//! — its tables' arguments, its stacked opening, its prepared openings. The
//! recursion splits the GROUPS across leaves: a group's opening covers all its
//! tables, so a group is the smallest unit a leaf can verify alone.
//!
//! # What a leaf does
//!
//! 1. The statement, as program constants ([`block_statement_bytes`]), then the
//!    roots block over EVERY root of the block — the groups' (arena) and the
//!    prepared tables' (constants the plan derived) — then `z, α, β` and the
//!    transcript's state digest. Every leaf replays the same front, so every leaf
//!    forks from the same `S_post`.
//! 2. Per group of its own: the fork, each table's argument with its
//!    preprocessed leg ([`emit_table_walk`]), the group's opening
//!    ([`emit_group_walk`]), each prepared table's opening
//!    ([`emit_prepared_group`]) — #1010's epoch legs, unchanged, on the fork.
//! 3. Publishes the block layout ([`BlockLayout::child`]): the block's id, the
//!    state, the public output's halves, and `Σ p/q` over its tables — minus the
//!    COMMIT-bus target on the leaf that carries it ([`CARRIER`]).
//!
//! The nodes are [`super::block_node`]'s, unchanged: they check the id, the
//! state and the output equal across children, add the sums, and the top
//! asserts the total is zero — the host's single bus check, where every table's
//! share is in scope. A leaf divides `p/q` itself, as #1010's epoch closure does,
//! so the node needs no fraction arithmetic ([`emit_share`]: the share is
//! `p · (1/q)`, so a zero denominator has no satisfying assignment whatever `p`
//! is, the host's `None`; `LFM_WHIR_SHARE_INVERSE=0` restores the former
//! `ediv(p, q)`, which leaves the share free when `p = q = 0`).
//!
//! # Never from a proof
//!
//! [`WhirBlockPlan::derive`] takes the statement's fields through the host
//! verifier's own checks ([`crate::block_whir::block_frame`]); every shape comes
//! from the AIR at the stated height, every count a leaf hints from that shape
//! ([`hint_table_wires_shaped`]), every derived root from the program; the leaf
//! partition and its carrier are the plan's. The proof is read only to fill a
//! leaf's arena — the witness.

use multilinear::stacking::StackedLayout;
use multilinear::whir::Domain;
use multilinear::whir_commit::Commitment;
use stark::multilinear_logup::InteractionShape;
use stark::multilinear_table::TableLayout;

use crate::block_whir::{
    BlockFormat, BlockFrame, BlockStatement, BlockWhirProof, block_frame, block_statement_bytes,
    commit_group, first_page_index, group_stacks, prepared_tables,
};
use crate::tables::types::{FE, FEE, GoldilocksExtension, GoldilocksField};

use super::block_node::{BlockLayout, BlockNodeInputs};
use super::builder::{Cell, Ext, LfmBuilder};
use super::compiler::{LfmProgram, compile};
use super::per_table_aggregator::DerivedChild;
use super::registry::LfmArtifacts;
use super::whir_chain::ChainShape;
use super::whir_epoch::{
    BITWISE_NAME, DECODE_NAME, GroupWires, KECCAK_RC_NAME, PreprocessedPlan, PreprocessedRoute,
    REGISTER_NAME, TableWires, emit_expected, emit_group_walk, emit_roots_block, emit_table_walk,
    fresh_schedule, group_columns, hint_group_chains_shaped, hint_table_wires_shaped,
    push_table_words, table_words,
};
use super::whir_stacked::{StackedPolyWires, stacked_verify_cost};
use super::whir_table::{TableProofWires, TableShape, table_verify_cost};
use super::whir_transcript::WhirTranscript;
use super::word::{LfmWord, base_word};

/// The leaf-load cap in permutations: today's wrap LFM_HASH shape, 2^18 + 2^15
/// rows, less the headroom wrap 2 runs at (D-NOEPOCH §12.2, the STARK block's
/// cap too).
pub const LEAF_PERMS_CAP: usize = 279_000;

/// The leaf that subtracts the COMMIT-bus target.
pub const CARRIER: usize = 0;

/// `LFM_WHIR_SHARE_INVERSE`: unset or `1` emits a leaf's bus share as
/// `p · (1/q)` ([`emit_share`]); `0` keeps the former `ediv(p, q)` leaf programs
/// (and their tree ids). Read once per process. Both sides derive the leaf
/// programs, so the verifier's setting is part of the tree identity it derives.
pub fn share_inverse() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(
        || match std::env::var("LFM_WHIR_SHARE_INVERSE").as_deref() {
            Err(_) | Ok("1") => true,
            Ok("0") => false,
            Ok(v) => panic!("LFM_WHIR_SHARE_INVERSE must be 0 or 1, got `{v}`"),
        },
    )
}

/// One table's share `p/q` of the block's bus sum. XALU division constrains
/// `q · out = p`, so `ediv(p, q)` has no satisfying assignment when `q = 0`
/// and `p ≠ 0`, but leaves `out` free when `p = q = 0`. With `inverse`, the
/// share is `p · ediv(1, q)`: `q · inv = 1` has no assignment for any `q = 0`,
/// whatever `p` is — the host's `contribution() = None` refusal.
pub fn emit_share(b: &mut LfmBuilder, p: Ext, q: Ext, inverse: bool) -> Ext {
    if inverse {
        let one = b.ext_const(&FEE::one());
        let inv = b.ediv(one, q);
        b.emul(p, inv)
    } else {
        b.ediv(p, q)
    }
}

/// Children a node verifies: three, so the block's three leaves close in ONE
/// node, the top, instead of two levels.
pub const BLOCK_FAN_IN: usize = 3;

// =============================== the partition ============================

/// Which groups each leaf verifies: lists of group indices that together cover
/// `0..num_groups` exactly once. (The no-epoch STARK block's `BlockPartition`,
/// over groups.)
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BlockPartition {
    leaves: Vec<Vec<usize>>,
    num_units: usize,
}

impl BlockPartition {
    /// Validate and sort each list. Refuses an empty leaf, an index out of
    /// range, an index in two leaves (or twice in one), and a unit in none.
    pub fn new(leaves: Vec<Vec<usize>>, num_units: usize) -> Result<Self, String> {
        let mut seen = vec![None::<usize>; num_units];
        let mut sorted = Vec::with_capacity(leaves.len());
        for (k, mut list) in leaves.into_iter().enumerate() {
            if list.is_empty() {
                return Err(format!("leaf {k} verifies nothing"));
            }
            list.sort_unstable();
            for &i in &list {
                let slot = seen
                    .get_mut(i)
                    .ok_or_else(|| format!("leaf {k}: {i} is out of range 0..{num_units}"))?;
                if let Some(other) = slot.replace(k) {
                    return Err(format!("{i} is in leaf {other} and leaf {k}"));
                }
            }
            sorted.push(list);
        }
        if let Some(i) = seen.iter().position(Option::is_none) {
            return Err(format!("{i} is in no leaf"));
        }
        Ok(Self {
            leaves: sorted,
            num_units,
        })
    }

    /// Lists taken as given, with none of [`Self::new`]'s checks: what an
    /// adversarial prover may emit leaves over. Only the tests that show such a
    /// tree is refused at the final check build one.
    #[cfg(test)]
    pub(crate) fn unvalidated(leaves: Vec<Vec<usize>>, num_units: usize) -> Self {
        Self { leaves, num_units }
    }

    pub fn num_leaves(&self) -> usize {
        self.leaves.len()
    }

    pub fn num_units(&self) -> usize {
        self.num_units
    }

    /// Leaf `k`'s units, ascending.
    pub fn leaf(&self, k: usize) -> &[usize] {
        &self.leaves[k]
    }

    pub fn leaves(&self) -> &[Vec<usize>] {
        &self.leaves
    }
}

/// The groups over `num_leaves` leaves: heaviest first, each onto the leaf with
/// the least load so far (ties: the lower group, then the lower leaf). Pure —
/// every emitter derives the same partition from the same costs — and never
/// more leaves than groups.
pub fn partition_groups(costs: &[usize], num_leaves: usize) -> BlockPartition {
    let k = num_leaves.clamp(1, costs.len().max(1));
    let mut order: Vec<usize> = (0..costs.len()).collect();
    order.sort_by_key(|&g| (std::cmp::Reverse(costs[g]), g));
    let mut leaves: Vec<Vec<usize>> = vec![Vec::new(); k];
    let mut load = vec![0usize; k];
    for g in order {
        let leaf = (0..k)
            .min_by_key(|&l| (load[l], l))
            .expect("at least one leaf");
        leaves[leaf].push(g);
        load[leaf] += costs[g];
    }
    BlockPartition::new(leaves, costs.len())
        .expect("the rule places every group exactly once by construction")
}

/// The leaf partition of groups costing `costs` permutations, under a leaf cap
/// of `cap`: over `num_leaves` leaves when the caller fixes the count (the
/// tests', never a statement's), else over the fewest leaves from
/// `⌈Σ cost / cap⌉` at which no leaf is over the cap — one more leaf while the
/// heaviest is over it, the STARK block's rule (`block_plan::partition_for`).
/// Refuses a group over the cap, which no leaf can hold: a group is atomic for
/// a leaf (its stacked opening covers all its tables), and without the bound a
/// statement could grow one leaf, and its LFM chips, without limit.
pub fn leaf_partition(
    costs: &[usize],
    num_leaves: Option<usize>,
    cap: usize,
) -> Result<BlockPartition, String> {
    if let Some((g, &cost)) = costs.iter().enumerate().find(|&(_, &c)| c > cap) {
        return Err(format!(
            "group {g} costs {cost} permutations — a leaf takes at most {cap}"
        ));
    }
    if let Some(k) = num_leaves {
        return Ok(partition_groups(costs, k.max(1)));
    }
    let mut k = costs.iter().sum::<usize>().div_ceil(cap).max(1);
    loop {
        let partition = partition_groups(costs, k);
        let heaviest = partition
            .leaves()
            .iter()
            .map(|leaf| leaf.iter().map(|&g| costs[g]).sum::<usize>())
            .max()
            .unwrap_or(0);
        // Every group is under the cap, so one group a leaf always fits.
        if heaviest <= cap || k >= costs.len() {
            return Ok(partition);
        }
        k += 1;
    }
}

// ================================= the plan ===============================

/// One group's prepared stack, as the plan derived it.
pub struct PlannedPrepared {
    pub group: usize,
    /// `(slot in the group's table order, AIR index, prefix length)`, in stack
    /// order.
    pub tables: Vec<(usize, usize, usize)>,
    /// Derived from the program and the partition, never read from a proof.
    pub roots: Vec<Commitment>,
    pub layout: StackedLayout,
    pub domain: Domain<GoldilocksField>,
}

impl PlannedPrepared {
    /// The prefix the stack settles of table `air`, when it holds it.
    pub fn settles(&self, air: usize) -> Option<usize> {
        self.tables
            .iter()
            .find(|&&(_, table, _)| table == air)
            .map(|&(_, _, n)| n)
    }
}

/// Every program of a block's tree, derived from the trusted ELF, the options,
/// the format and the statement. See the module docs.
pub struct WhirBlockPlan {
    frame: BlockFrame,
    statement_bytes: Vec<u8>,
    public_output: Vec<u8>,
    /// The statement's groups: AIR indices in each group's order.
    groups: Vec<Vec<usize>>,
    /// The prepared tables, in the proof's table order (the order their roots
    /// are absorbed and their openings carried).
    prepared: Vec<PlannedPrepared>,
    /// Each group's in-guest cost, in permutations.
    costs: Vec<usize>,
    partition: BlockPartition,
    /// Children a node verifies.
    fan_in: usize,
    /// A digest of the statement run: the block's identity, published by every
    /// leaf and node.
    id: [u8; 32],
}

/// Every program of a block's tree with its artifacts and derived shape, level
/// by level (the leaves first, the top last): what a prover proves and a
/// verifier derives, built from the plan alone.
pub struct TreePrograms {
    pub levels: Vec<Vec<TreeProgram>>,
}

pub struct TreeProgram {
    pub program: LfmProgram,
    pub artifacts: LfmArtifacts,
    pub derived: DerivedChild,
}

impl WhirBlockPlan {
    /// Derive the plan over `num_leaves` leaves (when `None`, the fewest from
    /// `⌈Σ cost / cap⌉` with no leaf over the cap: [`leaf_partition`]), or
    /// refuse a statement the host verifier refuses or whose group is over the
    /// leaf cap.
    pub fn derive(
        elf_bytes: &[u8],
        proof_options: &crate::ProofOptions,
        format: &BlockFormat,
        statement: BlockStatement<'_>,
        num_leaves: Option<usize>,
    ) -> Result<Self, String> {
        Self::derive_with(
            elf_bytes,
            proof_options,
            format,
            statement,
            num_leaves,
            BLOCK_FAN_IN,
            None,
        )
    }

    /// [`Self::derive`] with the nodes' fan-in named, and optionally the
    /// prepared tables' roots in `prepared_tables` order — a PROVER's own,
    /// computed by the same function from the same program, so it need not
    /// commit them twice. A verifier passes `None` and derives them itself;
    /// wrong roots give other leaf programs, and another top.
    pub fn derive_with(
        elf_bytes: &[u8],
        proof_options: &crate::ProofOptions,
        format: &BlockFormat,
        statement: BlockStatement<'_>,
        num_leaves: Option<usize>,
        fan_in: usize,
        prepared_roots: Option<&[(usize, Vec<Commitment>)]>,
    ) -> Result<Self, String> {
        Self::derive_capped(
            elf_bytes,
            proof_options,
            format,
            statement,
            num_leaves,
            fan_in,
            prepared_roots,
            LEAF_PERMS_CAP,
        )
    }

    /// [`Self::derive_with`] under a leaf cap of `leaf_cap` permutations
    /// ([`leaf_partition`]); everything but the tests' refusals runs at
    /// [`LEAF_PERMS_CAP`].
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn derive_capped(
        elf_bytes: &[u8],
        proof_options: &crate::ProofOptions,
        format: &BlockFormat,
        statement: BlockStatement<'_>,
        num_leaves: Option<usize>,
        fan_in: usize,
        prepared_roots: Option<&[(usize, Vec<Commitment>)]>,
        leaf_cap: usize,
    ) -> Result<Self, String> {
        if fan_in < 2 {
            return Err(format!("a fan-in of {fan_in} closes no tree"));
        }
        // The machine replays the algebraic transcript: it verifies a block
        // proved over RPX, and nothing else.
        if crate::whir_hash_knob::selected() != crate::whir_hash_knob::Setting::Rpx {
            return Err(
                "the block's recursion verifies a block proved over RPX; set LAMBDA_VM_WHIR_HASH=rpx"
                    .to_string(),
            );
        }
        let frame = block_frame(statement, elf_bytes, proof_options, format)
            .map_err(|e| format!("statement: {e:?}"))?;
        let elf_digest = crate::statement::elf_digest(elf_bytes);
        let statement_bytes = block_statement_bytes(statement, &elf_digest, &frame.config);
        let groups: Vec<Vec<usize>> = statement
            .groups
            .iter()
            .map(|g| g.iter().map(|&t| t as usize).collect())
            .collect();

        // The prepared tables: their layouts and domains from their shapes,
        // their roots committed here (or the prover's own, see
        // [`Self::derive_with`]).
        if !format.prepared {
            return Err(
                "the block's recursion needs the prepared openings (BlockFormat::prepared)"
                    .to_string(),
            );
        }
        let columns = prepared_tables(&frame.airs, &frame.page_configs, format)
            .map_err(|e| format!("prepared tables: {e:?}"))?;
        let config = &frame.config;
        let stacks = group_stacks(columns, statement.groups)
            .map_err(|e| format!("prepared stacks: {e:?}"))?;
        if let Some(roots) = prepared_roots
            && roots
                .iter()
                .map(|(g, _)| *g)
                .ne(stacks.iter().map(|s| s.group))
        {
            return Err("the supplied prepared roots are for other groups".to_string());
        }
        let mut prepared = Vec::with_capacity(stacks.len());
        for (index, stack) in stacks.into_iter().enumerate() {
            let group = stack.group;
            let start = groups[..group].iter().map(Vec::len).sum::<usize>();
            let tables: Vec<(usize, usize, usize)> = stack
                .tables
                .iter()
                .map(|&(position, air, n)| (position - start, air, n))
                .collect();
            let layout =
                stark::multilinear_table::global_layout(&stack.shapes(), config.format.stack)
                    .map_err(|e| format!("prepared group {group}: {e:?}"))?;
            let domain = Domain::<GoldilocksField>::new(layout.n_stack() + config.log_blowup)
                .map_err(|e| format!("prepared group {group}: {e:?}"))?;
            let roots = match prepared_roots {
                Some(roots) => roots[index].1.clone(),
                None => crate::with_whir_hash!(|H| {
                    commit_group::<H>(stack, config)
                        .map_err(|e| format!("prepared group {group}: {e:?}"))?
                        .roots
                }),
            };
            if roots.len() != layout.num_polys() {
                return Err(format!(
                    "prepared group {group}: {} roots for {} polynomials",
                    roots.len(),
                    layout.num_polys()
                ));
            }
            prepared.push(PlannedPrepared {
                group,
                tables,
                roots,
                layout,
                domain,
            });
        }

        // Each group's cost: its tables' legs, its opening, its prepared ones.
        let refs = frame.airs.air_refs();
        let mut costs = Vec::with_capacity(groups.len());
        for (g, list) in groups.iter().enumerate() {
            let owned = Shapes::build(&refs, &frame.shapes, list)?;
            let shapes = owned.table_shapes();
            let entry = fresh_schedule().entry();
            let tables: usize = match batched_cap(config) {
                None => shapes
                    .iter()
                    .map(|shape| table_verify_cost(shape, entry).perms())
                    .sum(),
                Some(cap) => {
                    super::whir_batch::batched_argue_cost(&shapes, &owned.argue_plan(cap), entry)
                        .perms()
                }
            };
            let group_shapes: Vec<(usize, usize)> = list.iter().map(|&t| frame.shapes[t]).collect();
            let (_, group_of) = group_columns(&group_shapes);
            let layout = &frame.stack_layouts[g];
            let chain = ChainShape::new(config, layout.n_stack());
            let opening = stacked_verify_cost(layout, &group_of, &chain, entry).perms();
            let prepared_perms: usize = prepared
                .iter()
                .filter(|p| p.group == g)
                .map(|p| {
                    let group_of: Vec<usize> = p
                        .tables
                        .iter()
                        .enumerate()
                        .flat_map(|(k, &(_, _, n))| std::iter::repeat_n(k, n))
                        .collect();
                    let chain = ChainShape::new(config, p.layout.n_stack());
                    stacked_verify_cost(&p.layout, &group_of, &chain, entry).perms()
                })
                .sum();
            costs.push(tables + opening + prepared_perms);
        }
        drop(refs);
        let partition = leaf_partition(&costs, num_leaves, leaf_cap)?;
        let id = crate::statement::elf_digest(&statement_bytes);
        Ok(Self {
            frame,
            statement_bytes,
            public_output: statement.public_output.to_vec(),
            groups,
            prepared,
            costs,
            partition,
            fan_in,
            id,
        })
    }

    pub fn partition(&self) -> &BlockPartition {
        &self.partition
    }

    /// Each group's in-guest cost, in permutations.
    pub fn costs(&self) -> &[usize] {
        &self.costs
    }

    pub fn carrier(&self) -> usize {
        CARRIER
    }

    pub fn num_groups(&self) -> usize {
        self.groups.len()
    }

    /// The prepared tables, in the proof's table order.
    pub fn prepared(&self) -> &[PlannedPrepared] {
        &self.prepared
    }

    /// The block's identity: a digest of the statement run.
    pub fn id(&self) -> &[u8; 32] {
        &self.id
    }

    /// What a leaf and a non-top node publish.
    pub fn child_layout(&self) -> BlockLayout {
        BlockLayout::child(STATE_WORDS, out_halves(&self.public_output).len())
    }

    /// What the top node publishes.
    pub fn top_layout(&self) -> BlockLayout {
        BlockLayout::top(STATE_WORDS, out_halves(&self.public_output).len())
    }

    /// Leaf `k`, emitted and validated.
    pub fn leaf_program(&self, k: usize) -> Result<LfmProgram, String> {
        if k >= self.partition.num_leaves() {
            return Err(format!("no leaf {k}"));
        }
        let mut b = builder();
        emit_block_leaf(&mut b, self, k)?;
        finish(b)
    }

    /// The node over `children`, the top when `top`, emitted and validated.
    pub fn node_program(
        &self,
        children: &[&DerivedChild],
        top: bool,
    ) -> Result<LfmProgram, String> {
        let shapes: Vec<_> = children.iter().map(|c| c.shape()).collect();
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

    /// The levels above the leaves: the plan's fan-in, leftovers in arity-1
    /// nodes, the last level's single node the top. A one-leaf block still gets
    /// a top node: the bus closes there.
    pub fn levels(&self) -> Vec<super::per_table_aggregator::Level> {
        use super::per_table_aggregator::{Level, tree_shape};
        let mut shape = tree_shape(self.partition.num_leaves(), self.fan_in);
        if shape.is_empty() {
            shape.push(Level { arities: vec![1] });
        }
        shape
    }

    /// Every program of the tree with its artifacts and derived shape, with no
    /// proof: every leaf emitted (in parallel) and its artifacts built, each
    /// node emitted over its children's derived shapes, level by level, under
    /// the tree's options.
    pub fn programs(&self, wrap_opts: &crate::ProofOptions) -> Result<TreePrograms, String> {
        let words = self.child_layout().total();
        let build = |program: LfmProgram| -> Result<TreeProgram, String> {
            let artifacts = artifacts_of(&program, wrap_opts);
            let derived = DerivedChild::from_artifacts(&artifacts, wrap_opts, words)?;
            Ok(TreeProgram {
                program,
                artifacts,
                derived,
            })
        };
        let leaf = |k: usize| build(self.leaf_program(k)?);
        #[cfg(feature = "parallel")]
        let first: Vec<TreeProgram> = {
            use rayon::prelude::*;
            (0..self.partition.num_leaves())
                .into_par_iter()
                .map(leaf)
                .collect::<Result<_, String>>()?
        };
        #[cfg(not(feature = "parallel"))]
        let first: Vec<TreeProgram> = (0..self.partition.num_leaves())
            .map(leaf)
            .collect::<Result<_, String>>()?;
        let mut levels = vec![first];
        let shape = self.levels();
        for (lv, arities) in shape.iter().enumerate() {
            let top = lv + 1 == shape.len();
            let below = levels.last().expect("the leaves");
            let mut at = 0usize;
            let mut next = Vec::with_capacity(arities.arities.len());
            for &a in &arities.arities {
                let kids = below
                    .get(at..at + a)
                    .ok_or_else(|| format!("level {}: arities overrun the children", lv + 1))?;
                let derived: Vec<&DerivedChild> = kids.iter().map(|k| &k.derived).collect();
                next.push(build(self.node_program(&derived, top)?)?);
                at += a;
            }
            if at != below.len() {
                return Err(format!("level {}: arities leave children over", lv + 1));
            }
            levels.push(next);
        }
        match levels.last().map(Vec::len) {
            Some(1) => Ok(TreePrograms { levels }),
            other => Err(format!("the tree closes to {other:?} nodes")),
        }
    }

    /// The top program's artifacts, with no proof: [`Self::derive_top_in`]
    /// with the leaves built in windows of `LAMBDA_VM_BLOCK_DERIVE_BUILDS`
    /// ([`derive_builds_bound`]), or under `LAMBDA_VM_BLOCK_DERIVE_HOLD=level`
    /// (the A/B control, [`derive_holds_levels`]) every program of the tree
    /// held to the end ([`Self::programs`]), as the verifier did before.
    pub fn derive_top(&self, wrap_opts: &crate::ProofOptions) -> Result<LfmArtifacts, String> {
        if derive_holds_levels()? {
            let mut programs = self.programs(wrap_opts)?;
            let top = programs
                .levels
                .pop()
                .and_then(|mut level| level.pop())
                .ok_or("no top")?;
            return Ok(top.artifacts);
        }
        self.derive_top_in(wrap_opts, derive_builds_bound()?)
    }

    /// The top program's artifacts, derived as a verifier needs them: the
    /// programs of [`Self::programs`], built in the same order, but each leaf
    /// and node keeps only its child's shape ([`DerivedChild`], what the node
    /// above is emitted over): its program and artifacts are dropped as soon
    /// as that shape is derived, and only the top's artifacts are kept. The
    /// leaves are built in windows of `window` (the whole level at once when
    /// `None`), each window in parallel; the nodes one at a time. Holding
    /// every level ([`Self::programs`]) took the p90 block's verifier to
    /// 41.17 GiB live (RYZEN 071, I-MEMFIT §6.24). Scheduling and retention
    /// only: every artifact is a pure function of its program and the
    /// options, so the top is [`Self::programs`]' top.
    pub fn derive_top_in(
        &self,
        wrap_opts: &crate::ProofOptions,
        window: Option<usize>,
    ) -> Result<LfmArtifacts, String> {
        let words = self.child_layout().total();
        let derive = |program: LfmProgram| -> Result<DerivedChild, String> {
            let artifacts = artifacts_of(&program, wrap_opts);
            DerivedChild::from_artifacts(&artifacts, wrap_opts, words)
        };
        let leaves: Vec<usize> = (0..self.partition.num_leaves()).collect();
        let mut below = map_in_windows(&leaves, window, |&k| derive(self.leaf_program(k)?))?;
        let shape = self.levels();
        for (lv, arities) in shape.iter().enumerate() {
            let top = lv + 1 == shape.len();
            let mut at = 0usize;
            let mut next = Vec::with_capacity(arities.arities.len());
            for &a in &arities.arities {
                let kids: Vec<&DerivedChild> = below
                    .get(at..at + a)
                    .ok_or_else(|| format!("level {}: arities overrun the children", lv + 1))?
                    .iter()
                    .collect();
                let program = self.node_program(&kids, top)?;
                if top {
                    if arities.arities.len() != 1 {
                        return Err(format!(
                            "the tree closes to {:?} nodes",
                            Some(arities.arities.len())
                        ));
                    }
                    let artifacts = artifacts_of(&program, wrap_opts);
                    // The top's shape too, as [`Self::programs`] derives it: the
                    // same refusals.
                    DerivedChild::from_artifacts(&artifacts, wrap_opts, words)?;
                    if at + a != below.len() {
                        return Err(format!("level {}: arities leave children over", lv + 1));
                    }
                    return Ok(artifacts);
                }
                next.push(derive(program)?);
                at += a;
            }
            if at != below.len() {
                return Err(format!("level {}: arities leave children over", lv + 1));
            }
            below = next;
        }
        Err("no top".to_string())
    }

    /// Replace the partition — another tree, test-only: what a prover may emit
    /// leaves over, refused at the final check.
    #[cfg(test)]
    pub(crate) fn with_partition(mut self, partition: BlockPartition) -> Self {
        assert_eq!(partition.num_units(), self.groups.len());
        self.partition = partition;
        self
    }

    /// Where group `g`'s first table sits in the proof's table order.
    fn group_start(&self, g: usize) -> usize {
        self.groups[..g].iter().map(Vec::len).sum()
    }

    /// Where group `g`'s first root sits among the proof's roots.
    fn root_start(&self, g: usize) -> usize {
        self.frame.stack_layouts[..g]
            .iter()
            .map(StackedLayout::num_polys)
            .sum()
    }
}

/// Words the transcript's state digest occupies: one, on the algebraic sponge.
const STATE_WORDS: usize = 1;

/// The public output as the statement's 32-bit halves, little-endian, the last
/// one zero-padded.
pub fn out_halves(bytes: &[u8]) -> Vec<FE> {
    super::keccak_host::pack_stream(bytes)
}

/// The two published words of the block's id: its eight little-endian `u32`
/// halves, four to a word.
pub fn id_words(id: &[u8; 32]) -> [LfmWord; 2] {
    let halves = super::keccak_host::pack_stream(id);
    [
        [halves[0], halves[1], halves[2], halves[3]],
        [halves[4], halves[5], halves[6], halves[7]],
    ]
}

/// `LAMBDA_VM_BLOCK_DERIVE_BUILDS=<n>`: the block verifier builds a tree's
/// leaves in windows of `n` ([`WhirBlockPlan::derive_top`]), each window in
/// parallel; unset, the whole level is one window: #1013's knob of the same
/// name, on the WHIR tree. Scheduling only: every
/// artifact is a pure function of its program and the options, so the derived
/// top does not depend on it. A value that is not a positive integer is an
/// error.
pub fn derive_builds_bound() -> Result<Option<usize>, String> {
    parse_derive_builds(std::env::var(DERIVE_BUILDS_ENV).ok().as_deref())
}

/// [`derive_builds_bound`]'s knob.
pub const DERIVE_BUILDS_ENV: &str = "LAMBDA_VM_BLOCK_DERIVE_BUILDS";

/// `LAMBDA_VM_BLOCK_DERIVE_HOLD=level`: [`WhirBlockPlan::derive_top`] holds
/// every program and its artifacts to the end ([`WhirBlockPlan::programs`]),
/// as the verifier did before (the A/B control; #1013's knob of the same
/// name). Unset or empty, each is dropped once its child's shape is derived.
/// Any other value is an error.
pub fn derive_holds_levels() -> Result<bool, String> {
    parse_derive_hold(std::env::var(DERIVE_HOLD_ENV).ok().as_deref())
}

/// [`derive_holds_levels`]' knob.
pub const DERIVE_HOLD_ENV: &str = "LAMBDA_VM_BLOCK_DERIVE_HOLD";

/// A value of [`DERIVE_HOLD_ENV`].
pub fn parse_derive_hold(value: Option<&str>) -> Result<bool, String> {
    match value.map(str::trim) {
        None | Some("") => Ok(false),
        Some("level") => Ok(true),
        Some(v) => Err(format!("{DERIVE_HOLD_ENV} must be `level`, got `{v}`")),
    }
}

/// A value of [`DERIVE_BUILDS_ENV`]: unset or empty is the whole level.
pub fn parse_derive_builds(value: Option<&str>) -> Result<Option<usize>, String> {
    match value.map(str::trim) {
        None | Some("") => Ok(None),
        Some(v) => v
            .parse::<usize>()
            .ok()
            .filter(|&n| n >= 1)
            .map(Some)
            .ok_or_else(|| format!("{DERIVE_BUILDS_ENV} must be a positive integer, got `{v}`")),
    }
}

/// `f` over `items` in order, in windows of `window` (all of them at once when
/// `None`), each window in parallel where there is a pool; the first error
/// stops it. At most `window` items are in `f` at once.
pub(crate) fn map_in_windows<T: Sync, R: Send>(
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

/// A tree program's artifacts, under the block hasher.
pub fn artifacts_of(program: &LfmProgram, wrap_opts: &crate::ProofOptions) -> LfmArtifacts {
    super::program_census::build_artifacts_counted(
        program,
        wrap_opts,
        crate::hash_pin::BLOCK_HASHER,
    )
}

fn builder() -> LfmBuilder {
    LfmBuilder::new().with_wrap_hash(super::edsl::WrapHash::production())
}

fn finish(b: LfmBuilder) -> Result<LfmProgram, String> {
    let program = compile(b.finish());
    super::validator::validate(&program).map_err(|e| format!("not admissible: {e:?}"))?;
    Ok(program)
}

/// Whether `words`, a top proof's published words, claim the block: the plan's
/// id and the public output, in the top layout. The state is bound across the
/// children by the nodes.
pub fn top_claims(plan: &WhirBlockPlan, words: &[(u32, LfmWord)]) -> bool {
    let layout = plan.top_layout();
    if words.len() != layout.total()
        || words
            .iter()
            .enumerate()
            .any(|(i, (at, _))| *at as usize != i)
    {
        return false;
    }
    let id = id_words(&plan.id);
    (0..2).all(|h| words[layout.id(h)].1 == id[h])
        && out_halves(&plan.public_output)
            .iter()
            .enumerate()
            .all(|(i, v)| words[layout.out_half(i)].1 == base_word(*v))
}

/// ★ The no-epoch WHIR block's verifier over its tree's top proof: derive the
/// plan and the top program from the trusted ELF and the statement, verify
/// `top` against that program, and check it claims the block.
///
/// Every preset is pinned here, none is a caller's: the base's proof options
/// ([`super::proof::block_base_options`]), the block format
/// ([`BlockFormat::production`]), the tree's options
/// ([`super::proof::aggregation_wrap_options`]), the plan's leaf rule and its
/// fan-in ([`BLOCK_FAN_IN`]). A caller who could pass weaker ones would choose
/// the programs the top is checked against.
pub fn verify_block_tree(
    elf_bytes: &[u8],
    statement: BlockStatement<'_>,
    top: &super::proof::LfmProof,
) -> Result<(), String> {
    verify_block_tree_with(
        elf_bytes,
        &super::proof::block_base_options(),
        &BlockFormat::production(),
        statement,
        None,
        BLOCK_FAN_IN,
        &super::proof::aggregation_wrap_options(),
        top,
    )
}

/// [`verify_block_tree`] under presets of the caller's: the fixture formats,
/// leaf counts and fan-ins the tests run at. Test-only.
#[cfg(test)]
#[allow(clippy::too_many_arguments)]
pub(crate) fn verify_block_tree_under(
    elf_bytes: &[u8],
    proof_options: &crate::ProofOptions,
    format: &BlockFormat,
    statement: BlockStatement<'_>,
    num_leaves: Option<usize>,
    fan_in: usize,
    wrap_opts: &crate::ProofOptions,
    top: &super::proof::LfmProof,
) -> Result<(), String> {
    verify_block_tree_with(
        elf_bytes,
        proof_options,
        format,
        statement,
        num_leaves,
        fan_in,
        wrap_opts,
        top,
    )
}

#[allow(clippy::too_many_arguments)]
fn verify_block_tree_with(
    elf_bytes: &[u8],
    proof_options: &crate::ProofOptions,
    format: &BlockFormat,
    statement: BlockStatement<'_>,
    num_leaves: Option<usize>,
    fan_in: usize,
    wrap_opts: &crate::ProofOptions,
    top: &super::proof::LfmProof,
) -> Result<(), String> {
    let plan = WhirBlockPlan::derive_with(
        elf_bytes,
        proof_options,
        format,
        statement,
        num_leaves,
        fan_in,
        None,
    )?;
    let artifacts = plan.derive_top(wrap_opts)?;
    if !super::proof::verify_against_artifacts(&artifacts, &top.proof, &top.public_words, wrap_opts)
    {
        return Err("the top proof does not verify against the derived top program".to_string());
    }
    if !top_claims(&plan, &top.public_words) {
        return Err("the top proof does not claim this block".to_string());
    }
    Ok(())
}

// ================================ the leaf ================================

/// Some tables' layouts and buses, owned, for [`TableShape`]s to borrow.
struct Shapes<'a> {
    layouts: Vec<TableLayout<'a, GoldilocksField, GoldilocksExtension>>,
    buses: Vec<Vec<InteractionShape<GoldilocksExtension>>>,
}

type DynAir<'a> = dyn stark::traits::AIR<
        Field = GoldilocksField,
        FieldExtension = GoldilocksExtension,
        PublicInputs = (),
    > + 'a;

impl<'a> Shapes<'a> {
    /// The layouts of `tables` (AIR indices) at their stated shapes.
    fn build(
        refs: &[&'a DynAir<'a>],
        shapes: &[(usize, usize)],
        tables: &[usize],
    ) -> Result<Self, String> {
        let mut layouts = Vec::with_capacity(tables.len());
        let mut buses = Vec::with_capacity(tables.len());
        for &t in tables {
            let air = refs[t];
            let (width, num_vars) = shapes[t];
            let layout = crate::multilinear_prove::layout_of(air, width, num_vars)
                .map_err(|e| format!("{}: {e:?}", air.name()))?;
            let slots = layout.slot_of().to_vec();
            let bus = stark::multilinear_logup::interaction_shapes(
                air.bus_interactions(),
                slots.len(),
                |column| {
                    slots
                        .get(column)
                        .copied()
                        .ok_or(multilinear::Error::UnknownPolynomial {
                            index: column,
                            len: slots.len(),
                        })
                },
            )
            .map_err(|e| format!("{}: bus: {e:?}", air.name()))?;
            layouts.push(layout);
            buses.push(bus);
        }
        Ok(Self { layouts, buses })
    }

    fn table_shapes(&self) -> Vec<TableShape<'_>> {
        self.layouts
            .iter()
            .zip(&self.buses)
            .map(|(layout, bus)| TableShape {
                ir: layout.shape(),
                bus,
                kinds: layout.kinds(),
                num_columns: layout.num_columns(),
                num_vars: layout.num_vars(),
            })
            .collect()
    }

    /// The batched argue's plan over these tables, the host's own
    /// [`stark::multilinear_table::argue_plan`] at the format's cap: the bins
    /// are part of the transcript, so the leaf never restates them.
    fn argue_plan(&self, bin_log_cells: u8) -> stark::multilinear_table::ArguePlan {
        let shapes: Vec<stark::multilinear_table::ArgueShape> = self
            .layouts
            .iter()
            .map(|layout| stark::multilinear_table::ArgueShape::of(&layout.statement()))
            .collect();
        stark::multilinear_table::argue_plan(&shapes, bin_log_cells)
    }
}

/// The batched argue's bin cap when the block's format batches its groups.
fn batched_cap(config: &multilinear::whir_chain::ChainConfig) -> Option<u8> {
    match config.format.argue {
        multilinear::whir_chain::ArgueFormat::Batched { bin_log_cells } => Some(bin_log_cells),
        multilinear::whir_chain::ArgueFormat::PerTable => None,
    }
}

/// How a table's preprocessed columns are discharged in a leaf: the epoch's
/// routes by name, the page route for a page table, and nothing left for a
/// prepared table, whose opening settles every one of them. A table with
/// preprocessed columns and no route is refused, never skipped.
enum Route {
    None,
    Bitwise,
    ConstMle(Vec<Vec<FE>>),
    Page(Vec<Vec<FE>>),
    Prepared(usize),
}

fn route_of(
    plan: &WhirBlockPlan,
    air: &DynAir<'_>,
    table: usize,
    first_page: usize,
) -> Result<Route, String> {
    if let Some(n) = plan.prepared.iter().find_map(|p| p.settles(table)) {
        return Ok(Route::Prepared(n));
    }
    let pages = first_page..first_page + plan.frame.page_configs.len();
    if pages.contains(&table) {
        let config = &plan.frame.page_configs[table - first_page];
        let columns = if config.is_private_input {
            vec![crate::tables::page::offset_column()]
        } else {
            crate::tables::page::preprocessed_columns(config)
        };
        return Ok(Route::Page(columns));
    }
    let columns = air.precomputed_columns();
    if columns.is_empty() {
        return Ok(Route::None);
    }
    match air.name() {
        BITWISE_NAME => Ok(Route::Bitwise),
        KECCAK_RC_NAME | REGISTER_NAME => Ok(Route::ConstMle(columns)),
        DECODE_NAME => Err("DECODE is not prepared".to_string()),
        other => Err(format!(
            "table {table} ({other}) carries {} preprocessed columns and no route covers them",
            columns.len()
        )),
    }
}

/// What every leaf's front leaves: the carried roots, the transcript after the
/// draws, the challenges and the state digest.
struct Front {
    carried: Vec<Cell>,
    transcript: WhirTranscript,
    z: Ext,
    alpha: Ext,
    beta: Ext,
    state: Cell,
}

/// The front every leaf replays: every group's roots (hinted), the statement
/// (constants), the roots block with the derived prepared roots, `z, α, β`,
/// and the state digest (which does not advance the transcript).
fn emit_front(
    b: &mut LfmBuilder,
    plan: &WhirBlockPlan,
    arena: super::instr::ArenaId,
    at: &mut u32,
) -> Front {
    let num_roots: usize = plan
        .frame
        .stack_layouts
        .iter()
        .map(StackedLayout::num_polys)
        .sum();
    let carried: Vec<Cell> = (0..num_roots)
        .map(|_| {
            let cell = b.hint_word(arena, *at);
            *at += 1;
            cell
        })
        .collect();
    let mut transcript = WhirTranscript::new();
    transcript.absorb_const_bytes(&plan.statement_bytes);
    let derived: Vec<LfmWord> = plan
        .prepared
        .iter()
        .flat_map(|p| {
            p.roots
                .iter()
                .map(super::algebraic_commit::commitment_to_digest)
        })
        .collect();
    let (z, alpha, beta) = emit_roots_block(b, &mut transcript, &carried, &derived);
    let state = transcript.state(b);
    Front {
        carried,
        transcript,
        z,
        alpha,
        beta,
        state,
    }
}

/// The front alone, publishing `z, α, β` and then group `g`'s fork's first
/// draw — what a test compares against the host's transcript.
#[cfg(test)]
pub(crate) fn front_program(plan: &WhirBlockPlan, g: usize) -> LfmProgram {
    let mut b = builder();
    let arena = b.declare_arena(0);
    let mut at = 0u32;
    let front = emit_front(&mut b, plan, arena, &mut at);
    for e in [front.z, front.alpha, front.beta] {
        b.public(e.as_cell());
    }
    let mut fork = front.transcript.clone();
    fork.absorb_const_bytes(&(g as u64).to_le_bytes());
    let first = fork.sample_ext(&mut b);
    b.public(first.as_cell());
    b.set_arena_len(arena, at);
    compile(b.finish())
}

/// Which of a leaf's own checks it emits. Production emits all of them
/// ([`LeafChecks::ALL`]); a weakened set exists so a negative test can show the
/// check it names is the one refusing.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LeafChecks {
    /// The prepared openings (their chains are hinted either way, so the arena
    /// does not move).
    pub(crate) prepared: bool,
    /// Whether the carrier subtracts the COMMIT-bus target.
    pub(crate) target: bool,
}

impl LeafChecks {
    pub const ALL: Self = Self {
        prepared: true,
        target: true,
    };
}

/// ★ Emit block leaf `k`: the shared front, then its groups on their forks,
/// then its share of the bus and the block layout's publishes.
pub fn emit_block_leaf(b: &mut LfmBuilder, plan: &WhirBlockPlan, k: usize) -> Result<(), String> {
    emit_leaf(b, plan, k, LeafChecks::ALL)
}

/// [`emit_block_leaf`] under a weakened check set — a mutation, test-only.
#[cfg(test)]
pub(crate) fn leaf_program_with(
    plan: &WhirBlockPlan,
    k: usize,
    checks: LeafChecks,
) -> Result<LfmProgram, String> {
    let mut b = builder();
    emit_leaf(&mut b, plan, k, checks)?;
    finish(b)
}

fn emit_leaf(
    b: &mut LfmBuilder,
    plan: &WhirBlockPlan,
    k: usize,
    checks: LeafChecks,
) -> Result<(), String> {
    let frame = &plan.frame;
    let config = &frame.config;
    let refs = frame.airs.air_refs();
    let first_page = first_page_index(&frame.airs).map_err(|e| format!("{e:?}"))?;
    let arena = b.declare_arena(0);
    let mut at = 0u32;

    // ---- 1. the front.
    let Front {
        carried,
        transcript,
        z,
        alpha,
        beta,
        state,
    } = emit_front(b, plan, arena, &mut at);

    // ---- 2. the leaf's groups, each on its fork.
    let mut outputs: Vec<(Ext, Ext)> = Vec::new();
    for &g in plan.partition.leaf(k) {
        let tables = &plan.groups[g];
        let owned = Shapes::build(&refs, &frame.shapes, tables)?;
        let shapes = owned.table_shapes();
        let routes: Vec<Route> = tables
            .iter()
            .map(|&t| route_of(plan, refs[t], t, first_page))
            .collect::<Result<_, _>>()?;
        let views: Vec<Vec<&[FE]>> = routes
            .iter()
            .map(|route| match route {
                Route::ConstMle(columns) | Route::Page(columns) => {
                    columns.iter().map(Vec::as_slice).collect()
                }
                _ => Vec::new(),
            })
            .collect();
        let plans: Vec<PreprocessedPlan<'_>> = routes
            .iter()
            .zip(&views)
            .map(|(route, view)| match route {
                Route::None => PreprocessedPlan {
                    settled: 0,
                    route: PreprocessedRoute::None,
                },
                Route::Bitwise => PreprocessedPlan {
                    settled: 0,
                    route: PreprocessedRoute::Bitwise,
                },
                Route::ConstMle(_) => PreprocessedPlan {
                    settled: 0,
                    route: PreprocessedRoute::ConstMle(view),
                },
                Route::Page(_) => PreprocessedPlan {
                    settled: 0,
                    route: PreprocessedRoute::Page {
                        offset: view[0],
                        init: view.get(1).copied(),
                    },
                },
                Route::Prepared(settled) => PreprocessedPlan {
                    settled: *settled,
                    route: PreprocessedRoute::None,
                },
            })
            .collect();
        let slots: Vec<&[usize]> = owned.layouts.iter().map(|l| l.slot_of()).collect();
        let ladder_len = shapes
            .iter()
            .map(|shape| super::whir_bus::alpha_powers_read(shape.bus))
            .max()
            .unwrap_or(1);
        let ladder = super::whir_poly::emit_challenge_powers(b, alpha, ladder_len);

        let mut fork = transcript.clone();
        fork.absorb_const_bytes(&(g as u64).to_le_bytes());
        let walk = match batched_cap(config) {
            None => {
                let store: Vec<TableWires> = shapes
                    .iter()
                    .map(|shape| hint_table_wires_shaped(b, arena, &mut at, shape))
                    .collect();
                let wires: Vec<TableProofWires<'_>> =
                    store.iter().map(TableWires::borrow).collect();
                emit_table_walk(
                    b, &mut fork, &wires, &shapes, &plans, &slots, z, &ladder, beta,
                )
            }
            // The group's tables in one batched argue, its bins the host's.
            Some(cap) => {
                let argue_plan = owned.argue_plan(cap);
                let store =
                    super::whir_batch::hint_batched_shaped(b, arena, &mut at, &shapes, &argue_plan);
                super::whir_batch::emit_batched_walk(
                    b,
                    &mut fork,
                    &store,
                    &shapes,
                    &argue_plan,
                    &plans,
                    &slots,
                    z,
                    &ladder,
                    beta,
                )
            }
        };

        // The group's opening, against its own carried roots.
        let layout = &frame.stack_layouts[g];
        let chain = ChainShape::new(config, layout.n_stack());
        let held = hint_group_chains_shaped(b, arena, &mut at, layout.num_polys(), &chain);
        let openings: Vec<_> = held
            .storage
            .iter()
            .map(super::whir_chain::RoundStorage::openings)
            .collect();
        let rounds: Vec<Vec<super::whir_chain::ChainRoundWires<'_>>> = held
            .storage
            .iter()
            .zip(&openings)
            .map(|(chain, (current, next))| chain.wires(current, next))
            .collect();
        let root_at = plan.root_start(g);
        let polys: Vec<StackedPolyWires<'_>> = (0..held.finals.len())
            .map(|poly| StackedPolyWires {
                rounds: &rounds[poly],
                root: carried[root_at + poly],
                final_value: held.finals[poly],
            })
            .collect();
        emit_group_walk(
            b,
            &mut fork,
            &[GroupWires {
                layout,
                polys: &polys,
                shape: &chain,
                domain: &frame.domains[g],
            }],
            &[tables.len()],
            &walk,
        );

        // Its prepared stack, against the derived roots: each table's block at
        // that table's point.
        for p in plan.prepared.iter().filter(|p| p.group == g) {
            let shape = ChainShape::new(config, p.layout.n_stack());
            let held = hint_group_chains_shaped(b, arena, &mut at, p.layout.num_polys(), &shape);
            let roots: Vec<Cell> = p
                .roots
                .iter()
                .map(|root| {
                    b.digest_const(super::algebraic_commit::commitment_to_digest(root))
                        .as_cell()
                })
                .collect();
            let openings: Vec<_> = held
                .storage
                .iter()
                .map(super::whir_chain::RoundStorage::openings)
                .collect();
            let rounds: Vec<Vec<super::whir_chain::ChainRoundWires<'_>>> = held
                .storage
                .iter()
                .zip(&openings)
                .map(|(chain, (current, next))| chain.wires(current, next))
                .collect();
            let polys: Vec<StackedPolyWires<'_>> = (0..held.finals.len())
                .map(|poly| StackedPolyWires {
                    rounds: &rounds[poly],
                    root: roots[poly],
                    final_value: held.finals[poly],
                })
                .collect();
            if !checks.prepared {
                continue;
            }
            let mut points: Vec<&[Ext]> = Vec::new();
            let mut values: Vec<Ext> = Vec::new();
            for &(slot, _, n) in &p.tables {
                let column_at = walk.column_at(slot);
                points.extend(std::iter::repeat_n(walk.points[slot].as_slice(), n));
                values.extend_from_slice(&walk.values[column_at..column_at + n]);
            }
            super::whir_stacked::emit_stacked_verify(
                b, &mut fork, &p.layout, &polys, &points, &values, &shape, &p.domain,
            );
        }
        outputs.extend(walk.outputs.iter().copied());
    }

    // ---- 3. the leaf's share of the bus: Σ p/q, less the COMMIT-bus target on
    // the carrier (commits are indexed from 0 in a block).
    let mut sum: Option<Ext> = None;
    let inverse = share_inverse();
    for (p, q) in &outputs {
        let share = emit_share(b, *p, *q, inverse);
        sum = Some(match sum {
            None => share,
            Some(running) => b.eadd(running, share),
        });
    }
    let mut sum = sum.unwrap_or_else(|| b.ext_const(&FEE::zero()));
    if k == CARRIER && checks.target {
        let target = emit_expected(b, &plan.public_output, 0, z, alpha);
        sum = b.esub(sum, target);
    }

    // ---- the publishes, in `BlockLayout::child`'s order.
    for word in id_words(&plan.id) {
        let cell = b.digest_const(word).as_cell();
        b.public(cell);
    }
    b.public(state);
    for half in out_halves(&plan.public_output) {
        let felt = b.felt_const(half);
        b.public(felt.as_cell());
    }
    b.public(sum.as_cell());
    b.set_arena_len(arena, at);
    Ok(())
}

// ================================ the arena ===============================

/// Leaf `k`'s arena from the block proof — the witness — in the order
/// [`emit_block_leaf`] hints it. Refuses a proof whose argument does not have
/// the shapes the plan derived (a count the leaf would misread).
pub fn block_leaf_arena(
    plan: &WhirBlockPlan,
    proof: &BlockWhirProof,
    k: usize,
) -> Result<Vec<Vec<LfmWord>>, String> {
    let mut groups = Vec::with_capacity(plan.partition.leaf(k).len());
    for &g in plan.partition.leaf(k) {
        let start = plan.group_start(g);
        let tables = match batched_cap(&plan.frame.config) {
            None => proof
                .proof
                .tables
                .get(start..start + plan.groups[g].len())
                .ok_or("the proof is short of tables")?,
            Some(_) => &[],
        };
        let prepared = match plan.prepared.iter().position(|p| p.group == g) {
            Some(index) => Some(
                proof
                    .prepared
                    .get(index)
                    .ok_or("the proof is short of prepared openings")?,
            ),
            None => None,
        };
        let opening = proof
            .proof
            .columns
            .get(g)
            .ok_or("the proof is short of openings")?;
        groups.push(group_arena_words(
            plan,
            g,
            tables,
            proof.argues.get(g),
            opening,
            prepared,
        )?);
    }
    Ok(leaf_arena(&proof.proof.roots, groups))
}

/// A leaf's arena from the block's roots and its groups' words
/// ([`group_arena_words`]), in the leaf's group order.
pub fn leaf_arena(roots: &[Commitment], groups: Vec<Vec<LfmWord>>) -> Vec<Vec<LfmWord>> {
    let mut words: Vec<LfmWord> = roots
        .iter()
        .map(super::algebraic_commit::commitment_to_digest)
        .collect();
    for group in groups {
        words.extend(group);
    }
    vec![words]
}

/// Group `g`'s words of a leaf's arena, from its share of the proof alone: its
/// tables' proofs (the per-table format) or its batched argue, its opening and
/// its prepared opening. A group's words depend on nothing else, so a leaf can
/// be fed each group as phase B finishes it. Refuses the shapes the plan does
/// not derive, as [`block_leaf_arena`] does.
pub fn group_arena_words(
    plan: &WhirBlockPlan,
    g: usize,
    tables: &[stark::multilinear_table::TableProof<GoldilocksExtension>],
    argue: Option<&stark::multilinear_table::BatchedArgue<GoldilocksExtension>>,
    opening: &multilinear::stacked_eval::StackedProof<GoldilocksField, GoldilocksExtension>,
    prepared: Option<
        &multilinear::stacked_eval::StackedProof<GoldilocksField, GoldilocksExtension>,
    >,
) -> Result<Vec<LfmWord>, String> {
    let frame = &plan.frame;
    let refs = frame.airs.air_refs();
    let group_tables = plan.groups.get(g).ok_or("a group the plan does not hold")?;
    let owned = Shapes::build(&refs, &frame.shapes, group_tables)?;
    let mut words = Vec::new();
    match batched_cap(&frame.config) {
        None => {
            for (slot, shape) in owned.table_shapes().iter().enumerate() {
                let table = tables.get(slot).ok_or("the proof is short of tables")?;
                let before = words.len();
                push_table_words(&mut words, table);
                if words.len() - before != table_words(shape) {
                    return Err(format!(
                        "table {} carries {} words, its shape {}",
                        group_tables[slot],
                        words.len() - before,
                        table_words(shape)
                    ));
                }
            }
        }
        Some(cap) => {
            let argue = argue.ok_or("the proof is short of batched argues")?;
            let shapes = owned.table_shapes();
            let expected = super::whir_batch::batched_words(&shapes, &owned.argue_plan(cap));
            super::whir_batch::push_batched_words(&mut words, argue);
            if words.len() != expected {
                return Err(format!(
                    "group {g}'s batched argue carries {} words, its shapes {expected}",
                    words.len()
                ));
            }
        }
    }
    let layout = &frame.stack_layouts[g];
    let chain = ChainShape::new(&frame.config, layout.n_stack());
    push_chains(&mut words, opening, layout.num_polys(), &chain)?;
    let mut planned = plan.prepared.iter().filter(|p| p.group == g);
    match (planned.next(), prepared) {
        (Some(p), Some(opening)) => {
            let shape = ChainShape::new(&frame.config, p.layout.n_stack());
            push_chains(&mut words, opening, p.layout.num_polys(), &shape)?;
        }
        (Some(_), None) => return Err("the proof is short of prepared openings".to_string()),
        (None, Some(_)) => return Err(format!("a prepared opening for group {g}, which has none")),
        (None, None) => {}
    }
    if planned.next().is_some() {
        return Err(format!("group {g} plans more than one prepared stack"));
    }
    Ok(words)
}

/// One opening's chains: per polynomial its final value, then its rounds.
fn push_chains(
    words: &mut Vec<LfmWord>,
    opening: &multilinear::stacked_eval::StackedProof<GoldilocksField, GoldilocksExtension>,
    polys: usize,
    shape: &ChainShape,
) -> Result<(), String> {
    if opening.polys.len() != polys {
        return Err(format!(
            "an opening of {} chains where the layout has {polys}",
            opening.polys.len()
        ));
    }
    for chain in &opening.polys {
        words.push(super::word::ext_word(&chain.final_value));
        let before = words.len();
        super::whir_chain::push_round_words(words, shape, chain);
        if words.len() - before != super::whir_chain::RoundStorage::words(shape) as usize {
            return Err("a chain whose rounds are not its shape's".to_string());
        }
    }
    Ok(())
}
