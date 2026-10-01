//! A no-epoch BLOCK LEAF — the emitted verifier of a fixed subset of one block
//! proof's instances — and the partition that assigns every instance to
//! exactly one leaf.
//!
//! # What a leaf does (D-NOEPOCH §12.2)
//!
//! 1. The shared front ([`super::block_replay::replay_block_front`]): the
//!    monolithic statement and Phase A over ALL the block's roots, then `z, α`
//!    and the transcript's state digest.
//! 2. Per instance in its list: the fork, the challenge replay and today's
//!    per-table verification legs ([`super::epoch::emit_table_challenges`],
//!    [`super::epoch_verify::emit_table_verification`]) — the same functions
//!    every epoch wrap runs, on the roots step 1 absorbed.
//! 3. Publishes [`super::block_node::BlockLayout::child`]: the attestation id,
//!    the state digest, the public output halves, and `c` = the sum of its
//!    instances' bus contributions — minus the COMMIT-bus target on the leaf that
//!    carries it, so the sum over every leaf is zero exactly when the block's bus
//!    balances.
//!
//! What a leaf CANNOT check is that the other leaves forked from the same state,
//! verified the other instances, and summed the rest of the bus: those are the
//! nodes' checks ([`super::block_node`]).
//!
//! # Why the instance list is a constant
//!
//! A fork appends its instance index as a program constant
//! ([`super::epoch::fork_table`]), so a leaf's `program_id` commits to its list,
//! and a node absorbs each child's id as an emit-time constant. The partition is
//! therefore fixed when the tree's programs are emitted and is never read from a
//! proof; [`BlockPartition::new`] refuses a list set that misses or repeats an
//! instance, which is the only place coverage can go wrong.

use stark::config::Commitment;

use crate::tables::types::{FE, FEE};

use super::block_replay::{BlockStatementShape, replay_block_front};
use super::builder::{Ext, LfmBuilder};
use super::constraints::Analysis;
use super::epoch::{RootCells, TableAbsorbs, TableChallengeShape, fork_table};
use super::epoch_verify::{TableQueryArenas, TableVerifyShape};
use super::instr::ArenaId;
use super::statement_replay::{PhaseAPreprocessed, PhaseATable};

// =============================== the partition ============================

/// Which instances each leaf verifies: lists of instance indices that together
/// cover `0..num_instances` exactly once.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BlockPartition {
    leaves: Vec<Vec<usize>>,
    num_instances: usize,
}

impl BlockPartition {
    /// Validate and sort each list. Refuses an empty leaf, an index out of
    /// range, an index in two leaves (or twice in one), and an instance in none.
    pub fn new(leaves: Vec<Vec<usize>>, num_instances: usize) -> Result<Self, String> {
        let mut seen = vec![None::<usize>; num_instances];
        let mut sorted = Vec::with_capacity(leaves.len());
        for (k, mut list) in leaves.into_iter().enumerate() {
            if list.is_empty() {
                return Err(format!("leaf {k} verifies no instance"));
            }
            list.sort_unstable();
            for &i in &list {
                let slot = seen.get_mut(i).ok_or_else(|| {
                    format!("leaf {k}: instance {i} is out of range 0..{num_instances}")
                })?;
                if let Some(other) = slot.replace(k) {
                    return Err(format!("instance {i} is in leaf {other} and leaf {k}"));
                }
            }
            sorted.push(list);
        }
        if let Some(i) = seen.iter().position(Option::is_none) {
            return Err(format!("instance {i} is in no leaf"));
        }
        Ok(Self {
            leaves: sorted,
            num_instances,
        })
    }

    pub fn num_leaves(&self) -> usize {
        self.leaves.len()
    }

    pub fn num_instances(&self) -> usize {
        self.num_instances
    }

    /// Leaf `k`'s instances, ascending.
    pub fn leaf(&self, k: usize) -> &[usize] {
        &self.leaves[k]
    }

    pub fn leaves(&self) -> &[Vec<usize>] {
        &self.leaves
    }
}

/// D-NOEPOCH §12.2's rule over the block's AIR-order instance list:
///
/// 1. the KECCAK_RND instances seed leaves `0, 1, 2, …` (mod `num_leaves`);
/// 2. ECDAS, ECSM and KECCAK go to leaves 1, 2, 3 (mod `num_leaves`);
/// 3. the fixed and tiny tables (BITWISE, DECODE, KECCAK_RC, REGISTER, HALT,
///    COMMIT, HINT) go to leaf 0;
/// 4. every other instance, in AIR order but PAGE last, goes to the leaf whose
///    cost so far is smallest (ties: the lowest leaf).
///
/// `names` are the AIRs' names (`CPU[3]`, `PAGE:0x1000` …; the instance suffix
/// after `[` or `:` is ignored) and `costs` any additive per-instance cost.
/// Pure: every emitter that needs the partition derives the same one from the
/// same instance list.
pub fn partition_by_rule(names: &[&str], costs: &[usize], num_leaves: usize) -> BlockPartition {
    assert_eq!(names.len(), costs.len(), "one cost per instance");
    assert!(num_leaves >= 1, "a block has at least one leaf");
    let kind = |name: &str| -> String { name.split(['[', ':']).next().unwrap_or(name).to_string() };
    let mut leaves: Vec<Vec<usize>> = vec![Vec::new(); num_leaves];
    let mut placed = vec![false; names.len()];
    let mut place = |leaves: &mut Vec<Vec<usize>>, k: usize, i: usize| {
        leaves[k % num_leaves].push(i);
        placed[i] = true;
    };
    let mut keccak_rnd = 0usize;
    for (i, name) in names.iter().enumerate() {
        match kind(name).as_str() {
            "KECCAK_RND" => {
                place(&mut leaves, keccak_rnd, i);
                keccak_rnd += 1;
            }
            "ECDAS" => place(&mut leaves, 1, i),
            "ECSM" => place(&mut leaves, 2, i),
            "KECCAK" => place(&mut leaves, 3, i),
            "BITWISE" | "DECODE" | "KECCAK_RC" | "REGISTER" | "HALT" | "COMMIT" | "HINT" => {
                place(&mut leaves, 0, i)
            }
            _ => {}
        }
    }
    let rest: Vec<usize> = (0..names.len())
        .filter(|&i| !placed[i] && kind(names[i]) != "PAGE")
        .chain((0..names.len()).filter(|&i| !placed[i] && kind(names[i]) == "PAGE"))
        .collect();
    let mut load: Vec<usize> = leaves
        .iter()
        .map(|l| l.iter().map(|&i| costs[i]).sum())
        .collect();
    for i in rest {
        let k = (0..num_leaves)
            .min_by_key(|&k| (load[k], k))
            .expect("at least one leaf");
        leaves[k].push(i);
        load[k] += costs[i];
    }
    BlockPartition::new(leaves, names.len())
        .expect("the rule places every instance exactly once by construction")
}

// ================================ the leaf ================================

/// One instance a leaf verifies, as its emitter needs it. Every field is
/// program shape, derived from the AIR and the proof options.
pub struct BlockInstance<'a> {
    pub challenge: &'a TableChallengeShape,
    pub verify: &'a TableVerifyShape,
    pub analysis: &'a Analysis,
}

/// Everything a leaf's program is a function of.
pub struct BlockLeafInputs<'a> {
    pub statement: &'a BlockStatementShape,
    /// The trusted ELF's digest — absorbed as program text (host attestation).
    pub elf_digest: &'a [u8; 32],
    /// Per instance of the WHOLE block, in AIR order: the preprocessed root the
    /// verifier takes from the AIR (`air.precomputed_commitment_for(layout)`),
    /// when the instance is preprocessed. Program text, never proof data.
    pub precomputed_roots: &'a [Option<Commitment>],
    pub partition: &'a BlockPartition,
    /// Which of the partition's leaves this is.
    pub leaf: usize,
    /// One per index of `partition.leaf(leaf)`, in that order.
    pub instances: &'a [BlockInstance<'a>],
    /// The attestation id over the ELF's inputs
    /// (`programs::AttestedInputs::program_id`), published as two constant
    /// words in the wraps' layout.
    pub program_id: &'a [u8; 32],
    /// Whether this leaf subtracts the COMMIT-bus target from its sum — exactly
    /// one leaf of a block does.
    pub carries_commit_target: bool,
}

struct InstanceArenas {
    aux_root: Option<ArenaId>,
    contribution: Option<ArenaId>,
    composition_root: ArenaId,
    ood_current: ArenaId,
    ood_next: ArenaId,
    parts: ArenaId,
    fri_roots: ArenaId,
    fri_coeffs: ArenaId,
    nonce: Option<ArenaId>,
    legs: TableQueryArenas,
}

/// Emit a block leaf: declare its arenas (the public output, every instance's
/// main root, then per verified instance its proof data and query arenas, in
/// that order), replay the shared front, verify its instances, publish.
pub fn emit_block_leaf(b: &mut LfmBuilder, inputs: &BlockLeafInputs<'_>) {
    let BlockLeafInputs {
        statement,
        elf_digest,
        precomputed_roots,
        partition,
        leaf,
        instances,
        program_id,
        carries_commit_target,
    } = *inputs;
    let n = partition.num_instances();
    let indices = partition.leaf(leaf);
    assert_eq!(
        precomputed_roots.len(),
        n,
        "one preprocessed-root slot per instance of the block"
    );
    assert_eq!(
        instances.len(),
        indices.len(),
        "one shape per verified index"
    );
    for (inst, &idx) in instances.iter().zip(indices) {
        assert_eq!(
            (inst.challenge.index, inst.challenge.num_tables),
            (idx, n),
            "an instance's fork is its position in the block"
        );
    }
    let per_root = RootCells::words_per_root(b);

    // ---- arenas, in declaration order.
    let a_out = b.declare_arena(statement.out_halves() as u32);
    let a_main = b.declare_arena(per_root * n as u32);
    let arenas: Vec<InstanceArenas> = instances
        .iter()
        .map(|inst| {
            let c = inst.challenge;
            InstanceArenas {
                aux_root: c.has_aux_root.then(|| b.declare_arena(per_root)),
                contribution: c.has_contribution.then(|| b.declare_arena(1)),
                composition_root: b.declare_arena(per_root),
                ood_current: b.declare_arena((c.ood_current_dims.0 * c.ood_current_dims.1) as u32),
                ood_next: b.declare_arena((c.ood_next_dims.0 * c.ood_next_dims.1) as u32),
                parts: b.declare_arena(c.num_parts as u32),
                fri_roots: b.declare_arena(per_root * c.fri.num_committed() as u32),
                fri_coeffs: b.declare_arena(c.fri.num_terminal_coeffs() as u32),
                nonce: (c.grinding_factor > 0).then(|| b.declare_arena(1)),
                legs: super::epoch_verify::declare_table_arenas(b, inst.verify),
            }
        })
        .collect();

    // ---- the shared front: the statement, Phase A over every root, z, α.
    let public_output: Vec<_> = (0..statement.out_halves() as u32)
        .map(|i| b.hint_felt(a_out, i))
        .collect();
    let main_cells: Vec<RootCells> = (0..n)
        .map(|i| RootCells::hint(b, a_main, per_root * i as u32))
        .collect();
    let main_lanes: Vec<Vec<_>> = main_cells.iter().map(RootCells::lanes_flat).collect();
    let phase_a: Vec<PhaseATable> = (0..n)
        .map(|i| PhaseATable {
            preprocessed_root: precomputed_roots[i]
                .as_ref()
                .map(PhaseAPreprocessed::Constant),
            main_root: &main_lanes[i][..],
        })
        .collect();
    let front = replay_block_front(b, statement, elf_digest, &public_output, &phase_a);

    // ---- one fork per verified instance, with the full verification legs.
    let mut contributions: Vec<Ext> = Vec::new();
    for ((inst, a), &idx) in instances.iter().zip(&arenas).zip(indices) {
        let c = inst.challenge;
        let aux = a.aux_root.map(|id| RootCells::hint(b, id, 0));
        let contribution = a.contribution.map(|id| b.hint_word(id, 0).as_ext());
        let composition = RootCells::hint(b, a.composition_root, 0);
        let ood_current: Vec<Ext> = (0..(c.ood_current_dims.0 * c.ood_current_dims.1) as u32)
            .map(|k| b.hint_word(a.ood_current, k).as_ext())
            .collect();
        let ood_next: Vec<Ext> = (0..(c.ood_next_dims.0 * c.ood_next_dims.1) as u32)
            .map(|k| b.hint_word(a.ood_next, k).as_ext())
            .collect();
        let parts: Vec<Ext> = (0..c.num_parts as u32)
            .map(|k| b.hint_word(a.parts, k).as_ext())
            .collect();
        let fri_roots: Vec<RootCells> = (0..c.fri.num_committed())
            .map(|k| RootCells::hint(b, a.fri_roots, per_root * k as u32))
            .collect();
        let fri_coeffs: Vec<Ext> = (0..c.fri.num_terminal_coeffs() as u32)
            .map(|k| b.hint_word(a.fri_coeffs, k).as_ext())
            .collect();
        let nonce = a.nonce.map(|id| b.hint_felt(id, 0));
        if let Some(l) = contribution {
            contributions.push(l);
        }

        let mut fork = fork_table(&front.transcript, c.index, c.num_tables);
        let absorbs = TableAbsorbs {
            aux_root: aux.as_ref(),
            contribution,
            composition_root: &composition,
            ood_current: &ood_current,
            ood_next: &ood_next,
            parts: &parts,
            fri_roots: &fri_roots,
            fri_coeffs: &fri_coeffs,
            nonce,
        };
        let ch = super::epoch::emit_table_challenges(b, &mut fork, c, &absorbs);
        // The preprocessed root Phase A absorbed, as the same program text.
        let prep = precomputed_roots[idx].map(|r| RootCells::constant(b, &r));
        super::epoch_verify::emit_table_verification(
            b,
            inst.verify,
            inst.analysis,
            &ch,
            &absorbs,
            &super::epoch_verify::TableInputs {
                precomputed_root: prep.as_ref(),
                main_root: &main_cells[idx],
                rap_challenges: &[front.z, front.alpha],
            },
            &a.legs,
        );
    }

    // ---- the leaf's share of the bus.
    let mut sum = match contributions.split_first() {
        Some((first, rest)) => rest.iter().fold(*first, |acc, l| b.eadd(acc, *l)),
        None => b.ext_const(&FEE::zero()),
    };
    if carries_commit_target {
        // Monolithic proof: commits are indexed from 0.
        let start = b.felt_const(FE::from(0u64));
        let shape = super::logup::LogUpShape {
            num_contributing_tables: contributions.len(),
            num_output_bytes: statement.public_output_len,
        };
        let bytes = super::epoch::emit_output_bytes(b, &public_output, statement.public_output_len);
        let target =
            super::logup::emit_commit_bus_target(b, &shape, front.z, front.alpha, start, &bytes);
        sum = b.esub(sum, target);
    }

    // ---- publishes, in `BlockLayout::child`'s order.
    for word in super::programs::program_id_words(program_id) {
        let cell = b.digest_const(word).as_cell();
        b.public(cell);
    }
    for cell in front.state.cells() {
        b.public(*cell);
    }
    for half in &public_output {
        b.public(half.as_cell());
    }
    b.public(sum.as_cell());
}
