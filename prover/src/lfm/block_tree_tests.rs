//! The no-epoch block's recursion (D-NOEPOCH §12): the shared front against the
//! host transcript, the partition, the node bindings with a mutation per check,
//! block leaves over a real block proof, and the box-tier trees.
//!
//! Tiers:
//! - laptop: the front (synthetic statement and roots), the partition, the
//!   bindings over hinted words — programs of a few thousand instructions,
//!   executed, never proved;
//! - box, fixture scale: block leaves over a real monolithic proof of a small
//!   guest, proved and verified, the nodes above them, and the negatives that need
//!   real child proofs;
//! - box, production scale: the whole block, base to top node, timed.

use stark::config::Commitment;
use stark::proof::view::MultiProofView;

use crate::tables::types::{FE, FEE, GoldilocksExtension, GoldilocksField};

use super::block_leaf::{BlockPartition, emit_block_leaf_over, partition_by_rule};
use super::block_node::{
    BlockBindings, BlockLayout, BlockNodeInputs, bind_and_publish_with, emit_block_node_with,
};
use super::block_plan::{BlockShape, BlockTreePlan, le_halves, top_claims};
use super::block_replay::{BlockStatementShape, replay_block_front};
use super::builder::LfmBuilder;
use super::compiler::{LfmProgram, compile};
use super::edsl::WrapHash;
use super::epoch_tests::HostTable;
use super::epoch_verify_tests::TableLegs;
use super::executor::execute;
use super::per_table_aggregator::{DerivedChild, LegCells, hint_public_words, publics_arena};
use super::per_table_aggregator_tests::{RealChild, child_arena_words, child_shape};
use super::statement_replay::{NUM_TABLE_COUNTS, PhaseAPreprocessed, PhaseATable};
use super::word::{LfmWord, base_word, ext_word, word_as_ext};

type Gl = GoldilocksField;
type Ext3 = GoldilocksExtension;

fn production_builder() -> LfmBuilder {
    LfmBuilder::new().with_wrap_hash(WrapHash::production())
}

/// Bytes as the arena's `u32` halves: four bytes each, little-endian, the last
/// zero-padded.
/// The host transcript every block prover and verifier starts from:
/// `StatementKind::Monolithic` over the block's statement fields.
fn block_seed(
    elf_digest: &[u8; 32],
    public_output: &[u8],
    table_counts: &crate::TableCounts,
    num_private_input_pages: usize,
    runtime_page_ranges: &[crate::RuntimePageRange],
    fri_final_poly_log_degree: u8,
) -> crate::hash_pin::BlockTranscript {
    let mut t = crate::hash_pin::block_transcript(&[]);
    crate::statement::absorb_statement_with_digest(
        &mut t,
        crate::statement::StatementKind::Monolithic,
        elf_digest,
        public_output,
        table_counts,
        num_private_input_pages,
        runtime_page_ranges,
        fri_final_poly_log_degree,
    );
    t
}

// ================================ (a) the front ===========================

/// ★ (a) The machine's block front draws production's `z, α` and reaches the
/// host transcript's state, from the same statement and the same roots.
///
/// Synthetic data on purpose: a public output whose length is not a multiple of
/// four, distinct counts in declaration order, two page ranges, and preprocessed
/// roots on some tables only — every loop and branch of the absorb sequence. A
/// coalesced or miscounted call moves every value below it.
#[test]
fn the_block_front_replays_the_hosts_transcript() {
    use crypto::fiat_shamir::is_transcript::IsTranscript;

    assert_eq!(
        WrapHash::production(),
        WrapHash::Algebraic,
        "the block transcript is algebraic, and this compares its state word"
    );
    const OUTPUT_LEN: usize = 37;
    let elf_digest: [u8; 32] = core::array::from_fn(|i| 0x31 ^ (i as u8).wrapping_mul(7));
    let public_output: Vec<u8> = (0..OUTPUT_LEN)
        .map(|i| 0x5a ^ (i as u8).wrapping_mul(13))
        .collect();
    let page_ranges = [
        crate::RuntimePageRange {
            base: 0x2000,
            count: 2,
        },
        crate::RuntimePageRange {
            base: 0xa000,
            count: 1,
        },
    ];
    let counts = crate::TableCounts {
        cpu: 3,
        lt: 4,
        memw: 5,
        memw_aligned: 6,
        load: 7,
        mul: 8,
        dvrm: 9,
        shift: 10,
        branch: 11,
        memw_register: 12,
        eq: 13,
        bytewise: 14,
        store: 15,
        cpu32: 16,
        keccak: 1,
        keccak_rnd: 4,
        ecsm: 1,
        ecdas: 0,
        hint: 1,
        commit: 1,
        blake3: 0,
    };
    // An independent statement of the encoding, not `table_count_values`.
    let count_array: [u64; NUM_TABLE_COUNTS] = [
        3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 1, 4, 1, 0, 1, 1, 0,
    ];
    const PAGES: usize = 3;
    const FPLD: u8 = 5;
    let root = |k: u64| -> Commitment {
        super::algebraic_commit::digest_to_commitment(&[
            FE::from(k),
            FE::from(k.wrapping_mul(0x9e37_79b9)),
            FE::from(k + 17),
            FE::from(k ^ 0x5555),
        ])
    };
    const N: usize = 7;
    let prep: Vec<Option<Commitment>> = (0..N as u64)
        .map(|i| (i % 3 == 0).then(|| root(1000 + i)))
        .collect();
    let main: Vec<Commitment> = (0..N as u64).map(|i| root(2000 + i)).collect();

    // HOST.
    let mut host = block_seed(
        &elf_digest,
        &public_output,
        &counts,
        PAGES,
        &page_ranges,
        FPLD,
    );
    for i in 0..N {
        if let Some(p) = &prep[i] {
            host.append_bytes(p);
        }
        host.append_bytes(&main[i]);
    }
    let z = host.sample_field_element();
    let alpha = host.sample_field_element();
    let state = host.state_word();

    // MACHINE.
    let shape = BlockStatementShape {
        public_output_len: OUTPUT_LEN,
        table_counts: count_array,
        num_private_input_pages: PAGES as u64,
        fri_final_poly_log_degree: FPLD,
        page_ranges: page_ranges.iter().map(|r| (r.base, r.count)).collect(),
    };
    let mut b = production_builder();
    let a_out = b.declare_arena(shape.out_halves() as u32);
    let per_root = super::epoch::RootCells::words_per_root(&b);
    let a_main = b.declare_arena(per_root * N as u32);
    let out: Vec<_> = (0..shape.out_halves() as u32)
        .map(|i| b.hint_felt(a_out, i))
        .collect();
    let cells: Vec<_> = (0..N)
        .map(|i| super::epoch::RootCells::hint(&mut b, a_main, per_root * i as u32))
        .collect();
    let lanes: Vec<Vec<_>> = cells
        .iter()
        .map(super::epoch::RootCells::lanes_flat)
        .collect();
    let tables: Vec<PhaseATable> = (0..N)
        .map(|i| PhaseATable {
            preprocessed_root: prep[i].as_ref().map(PhaseAPreprocessed::Constant),
            main_root: &lanes[i][..],
        })
        .collect();
    let front = replay_block_front(&mut b, &shape, &elf_digest, &out, &tables);
    b.public(front.z.as_cell());
    b.public(front.alpha.as_cell());
    for c in front.state.cells() {
        b.public(*c);
    }
    let program = compile(b.finish());
    let arenas = vec![
        le_halves(&public_output)
            .into_iter()
            .map(base_word)
            .collect(),
        super::proof_arena::commitments_to_arena(&main),
    ];
    let exec = execute(&program, &arenas, &crate::hash_pin::BLOCK_HASHER)
        .expect("the block front must execute");
    let got = |i: usize| word_as_ext(&exec.public_words[i].1).expect("an ext challenge");
    assert_eq!(got(0), z, "z must be production's");
    assert_eq!(got(1), alpha, "α must be production's");
    assert_eq!(
        exec.public_words[2].1, state,
        "the state digest must be the host transcript's after z, α"
    );
    println!(
        "BLOCK FRONT: {} instructions, z α and state = host over {N} tables ({} preprocessed)",
        program.instrs.len(),
        prep.iter().flatten().count()
    );
}

// ============================== (b) the partition =========================

/// ★ The partition refuses every list set that does not cover each instance
/// exactly once — the only place coverage can go wrong, since the lists are
/// program constants after this. Each arm fails if its check is deleted from
/// `BlockPartition::new`.
#[test]
fn the_partition_refuses_a_gap_an_overlap_and_an_empty_leaf() {
    assert!(BlockPartition::new(vec![vec![0, 2], vec![1, 3]], 4).is_ok());
    let missing = BlockPartition::new(vec![vec![0, 2], vec![3]], 4).unwrap_err();
    assert!(missing.contains("instance 1 is in no leaf"), "{missing}");
    let twice = BlockPartition::new(vec![vec![0, 1, 2], vec![2, 3]], 4).unwrap_err();
    assert!(
        twice.contains("instance 2 is in leaf 0 and leaf 1"),
        "{twice}"
    );
    let dup = BlockPartition::new(vec![vec![0, 1, 1], vec![2, 3]], 4).unwrap_err();
    assert!(dup.contains("instance 1 is in leaf 0 and leaf 0"), "{dup}");
    let range = BlockPartition::new(vec![vec![0, 1], vec![2, 4]], 4).unwrap_err();
    assert!(range.contains("out of range"), "{range}");
    let empty = BlockPartition::new(vec![vec![0, 1, 2, 3], vec![]], 4).unwrap_err();
    assert!(empty.contains("verifies no instance"), "{empty}");
}

/// The rule's seeds and its min-load fill, on a toy instance list.
#[test]
fn the_partition_rule_seeds_the_wide_tables_and_fills_by_load() {
    let names = [
        "BITWISE",
        "DECODE",
        "KECCAK_RC",
        "REGISTER",
        "HALT",
        "COMMIT[0]",
        "KECCAK[0]",
        "KECCAK_RND[0]",
        "KECCAK_RND[1]",
        "ECSM[0]",
        "ECDAS[0]",
        "HINT[0]",
        "CPU[0]",
        "CPU[1]",
        "PAGE:0x0",
        "MEMW_R[0]",
    ];
    let costs = [1, 1, 1, 1, 1, 1, 50, 60, 60, 40, 40, 1, 30, 30, 10, 30];
    let p = partition_by_rule(&names, &costs, 4).expect("four leaves fill");
    // Seeds: KECCAK_RND 0/1 → leaves 0/1, ECDAS → 1, ECSM → 2, KECCAK → 3,
    // the fixed and tiny tables → 0. Loads after seeding: 67, 100, 40, 50.
    // Fill in AIR order, PAGE last: CPU[0] → 2 (40 → 70), CPU[1] → 3 (50 → 80),
    // MEMW_R[0] → 0 (67 → 97), PAGE → 2 (70 → 80).
    assert_eq!(
        p.leaves(),
        &[
            vec![0, 1, 2, 3, 4, 5, 7, 11, 15],
            vec![8, 10],
            vec![9, 12, 14],
            vec![6, 13],
        ]
    );
    // One leaf takes everything; the seeds wrap modulo the leaf count.
    let one = partition_by_rule(&names, &costs, 1).expect("one leaf fills");
    assert_eq!(one.leaves(), &[(0..names.len()).collect::<Vec<_>>()]);
    // More leaves than the rule can fill: refused, never an empty leaf.
    assert!(partition_by_rule(&names, &costs, names.len() + 1).is_err());
}

// ============================ (M1) the plan's shape ========================

/// A block shape for `poc_rodata_commit` the host's pre-checks accept: one CPU
/// and one MEMW_R table, the ELF's pages, every trace 2^5 rows.
fn honest_fixture_shape(elf: &executor::elf::Elf) -> BlockShape {
    let counts = crate::TableCounts {
        cpu: 1,
        lt: 0,
        memw: 0,
        memw_aligned: 0,
        load: 0,
        mul: 0,
        dvrm: 0,
        shift: 0,
        branch: 0,
        memw_register: 1,
        eq: 0,
        bytewise: 0,
        store: 0,
        cpu32: 0,
        keccak: 0,
        keccak_rnd: 0,
        ecsm: 0,
        ecdas: 0,
        hint: 0,
        commit: 0,
        blake3: 0,
    };
    let pages = crate::tables::trace_builder::Traces::page_configs_from_elf(elf).len();
    let n = counts.total().expect("fits") + crate::FIXED_TABLE_COUNT + pages;
    BlockShape {
        table_counts: counts,
        runtime_page_ranges: Vec::new(),
        num_private_input_pages: 0,
        public_output_len: 4,
        trace_lengths: vec![32; n],
    }
}

/// ★ M1: the plan refuses every shape the host verifier refuses, before it
/// builds an AIR ([`super::block_plan::check_shape`], `verify_proof_parts`'
/// pre-checks): a non-chunked accelerator counted twice, a KECCAK_RND count the
/// trace lengths do not cover, more private-input pages than the bound, a
/// runtime page range over an ELF page, and a trace length that is not a power
/// of two. Each tamper is the only change to a shape the checks accept.
#[test]
fn the_plan_refuses_a_shape_the_host_refuses() {
    use super::block_plan::check_shape;
    let opts = fixture_block_options();
    let elf_bytes = crate::test_utils::asm_elf_bytes("poc_rodata_commit");
    let elf = executor::elf::Elf::load(&elf_bytes).expect("load the ELF");
    let honest = honest_fixture_shape(&elf);
    assert!(
        check_shape(&elf, &opts, &honest).is_ok(),
        "the honest shape passes, so each refusal below is its tamper's"
    );
    let elf_page = crate::tables::trace_builder::Traces::page_configs_from_elf(&elf)
        .first()
        .expect("the ELF has a page")
        .page_base;
    type Tamper = Box<dyn Fn(&mut BlockShape)>;
    let tampers: Vec<(&str, Tamper)> = vec![
        (
            "a non-chunked accelerator counted twice",
            Box::new(|s| {
                s.table_counts.keccak = 2;
                s.trace_lengths.extend([32, 32]);
            }),
        ),
        (
            "a KECCAK_RND count the trace lengths do not cover",
            Box::new(|s| s.table_counts.keccak_rnd = 3),
        ),
        (
            "more private-input pages than the bound",
            Box::new(|s| {
                s.num_private_input_pages = crate::tables::page::max_private_input_pages() + 1;
            }),
        ),
        (
            "a runtime page range over an ELF page",
            Box::new(move |s| {
                s.runtime_page_ranges.push(crate::RuntimePageRange {
                    base: elf_page,
                    count: 1,
                });
                s.trace_lengths.push(32);
            }),
        ),
        (
            "a trace length that is not a power of two",
            Box::new(|s| s.trace_lengths[0] = 48),
        ),
    ];
    for (what, tamper) in tampers {
        let mut shape = honest.clone();
        tamper(&mut shape);
        let refused = check_shape(&elf, &opts, &shape);
        assert!(refused.is_err(), "the plan must refuse {what}");
        assert!(
            BlockTreePlan::derive(&elf_bytes, &opts, &shape).is_err(),
            "the derivation refuses {what}"
        );
        println!("PLAN refuses {what}: {}", refused.err().unwrap_or_default());
    }
    // KECCAK_RND is chunked under the block's accelerator shape: a count the
    // trace lengths DO cover passes the pre-checks (the sub-proof count is its
    // bound, as in `verify_proof_parts`).
    let mut chunked = honest.clone();
    chunked.table_counts.keccak_rnd = 3;
    chunked.trace_lengths.extend([32, 32, 32]);
    assert!(check_shape(&elf, &opts, &chunked).is_ok());
}

// =============================== (c) the bindings =========================

/// Two children's published words under [`BlockLayout::child`], honest: the same
/// id, state and output, and shares that cancel.
fn honest_children(layout: &BlockLayout) -> Vec<Vec<LfmWord>> {
    let id = super::programs::program_id_words(&core::array::from_fn(|i| 3 + i as u8));
    let state: Vec<LfmWord> = (0..layout.state_words as u64)
        .map(|w| [FE::from(11 + w), FE::from(12), FE::from(13), FE::from(14)])
        .collect();
    let out: Vec<LfmWord> = (0..layout.out_halves as u64)
        .map(|i| base_word(FE::from(0x100 + i)))
        .collect();
    let share = FEE::new([FE::from(5u64), FE::from(6u64), FE::from(7u64)]);
    [share, -share]
        .iter()
        .map(|s| {
            let mut words = id.to_vec();
            words.extend(state.iter().copied());
            words.extend(out.iter().copied());
            words.push(ext_word(s));
            assert_eq!(words.len(), layout.total());
            words
        })
        .collect()
}

/// A node's binding step over HINTED child words — the checks and publishes
/// `emit_block_node` runs after verifying its children, with the verification
/// left out so each check can be driven on its own.
fn bindings_program(
    layout: &BlockLayout,
    children: usize,
    top: bool,
    checks: BlockBindings,
) -> LfmProgram {
    let mut b = production_builder();
    let arenas: Vec<_> = (0..children)
        .map(|_| b.declare_arena(8 * layout.total() as u32))
        .collect();
    let zero = b.ext_const(&FEE::zero());
    let legs: Vec<LegCells> = arenas
        .iter()
        .map(|&a| LegCells {
            publics: hint_public_words(&mut b, a, layout.total()),
            z_alpha: (zero, zero),
        })
        .collect();
    bind_and_publish_with(&mut b, &legs, layout, top, checks);
    compile(b.finish())
}

fn run_bindings(
    layout: &BlockLayout,
    words: &[Vec<LfmWord>],
    top: bool,
    checks: BlockBindings,
) -> Result<Vec<LfmWord>, String> {
    let program = bindings_program(layout, words.len(), top, checks);
    let arenas: Vec<Vec<LfmWord>> = words.iter().map(|w| publics_arena(w)).collect();
    execute(&program, &arenas, &crate::hash_pin::BLOCK_HASHER)
        .map(|e| e.public_words.iter().map(|(_, w)| *w).collect())
        .map_err(|e| format!("{e:?}"))
}

/// ★ (c)/(e) Every cross-child check refuses its tamper, and each is the one
/// doing the refusing: with that check removed (`BlockBindings::without`) the same
/// tampered words execute.
///
/// - `id`: a child attesting another ELF's inputs;
/// - `state`: a child that forked from another transcript — another block's
///   statement or roots (a leaf over another block's proof);
/// - `out`: a child claiming another public output;
/// - `bus`: shares that do not cancel at the top — a tampered contribution, or a
///   target subtracted twice or not at all.
#[test]
fn every_block_binding_refuses_its_tamper_and_is_load_bearing() {
    let layout = BlockLayout::child(super::proof_arena::words_per_root(), 3);
    let honest = honest_children(&layout);
    let top_words = run_bindings(&layout, &honest, true, BlockBindings::ALL)
        .expect("honest children bind at the top");
    assert_eq!(
        top_words.len(),
        BlockLayout::top(layout.state_words, 3).total()
    );
    assert_eq!(
        top_words[..],
        honest[0][..layout.total() - 1],
        "the top republishes the claim"
    );
    let mid = run_bindings(&layout, &honest, false, BlockBindings::ALL)
        .expect("honest children bind below the top");
    assert_eq!(
        word_as_ext(mid.last().expect("a sum")).expect("an ext sum"),
        FEE::zero(),
        "a non-top node publishes the children's summed shares"
    );

    let tampers: [(&str, usize); 4] = [
        ("id", layout.id(1)),
        ("state", layout.state(0)),
        ("out", layout.out_half(2)),
        ("bus", layout.sum()),
    ];
    for (check, at) in tampers {
        let mut words = honest.clone();
        words[1][at][0] += FE::from(1u64);
        let refused = run_bindings(&layout, &words, true, BlockBindings::ALL);
        assert!(
            refused.is_err(),
            "`{check}`: a tampered word must be refused by the full binding set"
        );
        let mutant = run_bindings(&layout, &words, true, BlockBindings::without(check));
        assert!(
            mutant.is_ok(),
            "`{check}`: with the check removed the tamper executes — otherwise another \
             check is doing the refusing and this one is not load-bearing: {mutant:?}"
        );
    }
    // Below the top the bus is not closed, so a share that does not cancel is
    // carried up rather than refused.
    let mut words = honest.clone();
    words[1][layout.sum()][0] += FE::from(1u64);
    assert!(run_bindings(&layout, &words, false, BlockBindings::ALL).is_ok());
}

// =============================== the block harvest ========================

/// One block proof, production-accepted, read for leaf emission.
///
/// `plan` is the verifier's: every shape and constant a leaf is emitted from,
/// derived from the ELF, the options and the shape the proof claims — never
/// from its data. The other fields are the proof's DATA, which the leaves'
/// arenas carry: every instance's main root, the shared state, and per instance
/// the fork's data ([`HostTable`]) and the legs' openings ([`TableLegs`]), each
/// checked against the AIR's shape as it is read.
pub(super) struct RealBlock {
    pub(super) plan: BlockTreePlan,
    pub(super) public_output: Vec<u8>,
    pub(super) main_roots: Vec<Commitment>,
    pub(super) tables: Vec<HostTable>,
    pub(super) legs: Vec<TableLegs>,
    /// The host transcript's state after `z, α`.
    pub(super) state: LfmWord,
    /// The COMMIT-bus target production computed.
    pub(super) expected_bus_balance: FEE,
}

impl RealBlock {
    pub(super) fn num_instances(&self) -> usize {
        self.plan.num_instances()
    }

    /// AIR names, in AIR order.
    pub(super) fn names(&self) -> Vec<&str> {
        self.plan
            .instances()
            .iter()
            .map(|i| i.name.as_str())
            .collect()
    }

    /// The rule over `num_leaves` leaves: a TEST partition, another tree than
    /// the plan's unless the count is the plan's own.
    pub(super) fn partition_over(&self, num_leaves: usize) -> BlockPartition {
        partition_by_rule(&self.names(), &self.plan.costs(), num_leaves)
            .expect("the rule fills every leaf")
    }

    /// What leaf `k` of `partition` must publish, from the host's own values:
    /// the id, the state, the output halves, and its share of the bus — less
    /// the target when it `carries` it.
    pub(super) fn expected_leaf_publics(
        &self,
        partition: &BlockPartition,
        k: usize,
        carries: bool,
    ) -> Vec<LfmWord> {
        let mut words =
            super::programs::program_id_words(&self.plan.attested().program_id()).to_vec();
        words.push(self.state);
        words.extend(le_halves(&self.public_output).into_iter().map(base_word));
        let mut share = partition
            .leaf(k)
            .iter()
            .filter_map(|&i| self.tables[i].contribution)
            .fold(FEE::zero(), |acc, l| acc + l);
        if carries {
            share = share - self.expected_bus_balance;
        }
        words.push(ext_word(&share));
        words
    }
}

/// Harvest a block proof that production's block verifier accepts.
///
/// The plan comes first ([`BlockTreePlan::derive`], from the shape the proof
/// claims), and the harvest verifies and reads the proof over the plan's own
/// AIR set — `verify_proof_parts`' set. Returns the harvest and the seconds of
/// its two halves — the plan and the host verify, which only refuses (a harness
/// assert, as in every tree driver), and the replay a driver needs.
pub(super) fn harvest_block(
    opts: &crate::ProofOptions,
    elf_bytes: &[u8],
    proof: &crate::VmProof,
) -> Result<(RealBlock, f64, f64), String> {
    harvest_block_with(opts, elf_bytes, proof, true)
}

/// [`harvest_block`], with the host verify run or skipped. Skipping it is for a
/// caller that runs the verifying harvest beside its own work and joins it
/// before it reports anything (`the_block_tree_composes_to_a_top_node`): the
/// assert still decides whether the run counts, it only stops gating the leaves.
pub(super) fn harvest_block_with(
    opts: &crate::ProofOptions,
    elf_bytes: &[u8],
    proof: &crate::VmProof,
    verify: bool,
) -> Result<(RealBlock, f64, f64), String> {
    use crypto::fiat_shamir::is_transcript::IsTranscript;
    use rayon::prelude::*;
    use stark::verifier::IsStarkVerifier;

    let t_verify = std::time::Instant::now();
    let plan = BlockTreePlan::derive(elf_bytes, opts, &BlockShape::of_proof(proof))?;
    let view = MultiProofView::Owned(&proof.proof);
    let refs = plan.airs().air_refs();
    let seed = || {
        block_seed(
            plan.elf_digest(),
            &proof.public_output,
            &proof.table_counts,
            proof.num_private_input_pages,
            &proof.runtime_page_ranges,
            opts.fri_final_poly_log_degree,
        )
    };
    let expected = crate::compute_expected_commit_bus_balance_view(
        &refs,
        view,
        &proof.public_output,
        0,
        &mut seed(),
    )
    .ok_or("the COMMIT bus target must compute")?;
    if verify
        && !crate::hash_pin::BlockVerifier::<Gl, Ext3, ()>::multi_verify_views(
            &refs,
            view,
            &mut seed(),
            &expected,
        )
    {
        return Err("production's verifier rejects the block".to_string());
    }
    let verify_secs = t_verify.elapsed().as_secs_f64();

    // ---- Phase A, as `multi_verify_views` absorbs it: the plan's preprocessed
    // roots (the AIR's, never the proof's), then the proof's main roots.
    let t_replay = std::time::Instant::now();
    let n = refs.len();
    let mut transcript = seed();
    let mut main_roots = Vec::with_capacity(n);
    for idx in 0..n {
        let v = view.get(idx);
        if let Some(p) = &plan.instance(idx).precomputed_root {
            transcript.append_bytes(p);
        }
        transcript.append_bytes(v.lde_trace_main_merkle_root());
        main_roots.push(*v.lde_trace_main_merkle_root());
    }
    let lookup: Vec<FEE> = (0..stark::lookup::LOGUP_NUM_CHALLENGES)
        .map(|_| transcript.sample_field_element())
        .collect();
    let state = transcript.state_word();

    // ---- one fork per instance, and the legs' reading of the same sub-proof.
    let per_instance: Vec<(HostTable, TableLegs)> = (0..n)
        .into_par_iter()
        .map(|idx| {
            let air = refs[idx];
            let v = view.get(idx);
            let mut fork = transcript.clone();
            if n > 1 {
                fork.append_bytes(&(idx as u64).to_le_bytes());
            }
            if let Some(root) = v.lde_trace_aux_merkle_root() {
                fork.append_bytes(root);
            }
            if let Some(c) = v.bus_table_contribution() {
                fork.append_field_element(&c);
            }
            let table = super::epoch_tests::host_table_forked(air, v, idx, n, &mut fork, &lookup);
            let legs = super::epoch_verify_tests::build_table_legs(air, v, &lookup);
            (table, legs)
        })
        .collect();
    let (tables, legs): (Vec<_>, Vec<_>) = per_instance.into_iter().unzip();
    // The proof-reading path and the plan derive one shape per instance.
    for (i, (t, l)) in tables.iter().zip(&legs).enumerate() {
        let planned = plan.instance(i);
        assert_eq!(
            t.precomputed_root, planned.precomputed_root,
            "instance {i}: the fork's and the plan's preprocessed roots are one value"
        );
        assert_eq!(
            format!("{:?}", t.shape),
            format!("{:?}", planned.challenge),
            "instance {i}: the plan's challenge shape is the one the proof is read against"
        );
        assert_eq!(
            format!("{:?}", l.verify),
            format!("{:?}", planned.verify),
            "instance {i}: the plan's legs shape is the one the proof is read against"
        );
    }
    let replay_secs = t_replay.elapsed().as_secs_f64();

    Ok((
        RealBlock {
            plan,
            public_output: proof.public_output.clone(),
            main_roots,
            tables,
            legs,
            state,
            expected_bus_balance: expected,
        },
        verify_secs,
        replay_secs,
    ))
}

/// Leaf `k` of the plan — the program the verifier derives.
pub(super) fn block_leaf_program(rb: &RealBlock, k: usize) -> LfmProgram {
    rb.plan
        .leaf_program(k)
        .unwrap_or_else(|e| panic!("leaf {k} must emit: {e}"))
}

/// Leaf `k` of another partition, carrying the COMMIT-bus target when
/// `carries` — a tree the verifier did not derive, for the negatives.
pub(super) fn block_leaf_program_over(
    rb: &RealBlock,
    partition: &BlockPartition,
    k: usize,
    carries: bool,
) -> LfmProgram {
    let mut b = production_builder();
    emit_block_leaf_over(&mut b, &rb.plan, partition, k, carries);
    let program = compile(b.finish());
    super::validator::validate(&program).expect("a block leaf must be admissible");
    program
}

/// Leaf `k`'s arenas, in `emit_block_leaf`'s declaration order.
pub(super) fn block_leaf_arenas(
    rb: &RealBlock,
    partition: &BlockPartition,
    k: usize,
) -> Vec<Vec<LfmWord>> {
    let mut arenas: Vec<Vec<LfmWord>> = vec![
        le_halves(&rb.public_output)
            .into_iter()
            .map(base_word)
            .collect(),
        super::proof_arena::commitments_to_arena(&rb.main_roots),
    ];
    for &i in partition.leaf(k) {
        let (h, leg) = (&rb.tables[i], &rb.legs[i]);
        if let Some(root) = &h.aux_root {
            arenas.push(super::proof_arena::commitments_to_arena(&[*root]));
        }
        if let Some(l) = &h.contribution {
            arenas.push(vec![ext_word(l)]);
        }
        arenas.push(super::proof_arena::commitments_to_arena(&[
            h.composition_root
        ]));
        arenas.push(h.ood_current.iter().map(ext_word).collect());
        arenas.push(h.ood_next.iter().map(ext_word).collect());
        arenas.push(h.parts.iter().map(ext_word).collect());
        arenas.push(super::proof_arena::commitments_to_arena(&h.fri_roots));
        arenas.push(h.fri_coeffs.iter().map(ext_word).collect());
        if let Some(nonce) = h.nonce {
            arenas.push(vec![base_word(FE::from(nonce))]);
        }
        arenas.push(leg.opening_arena());
        arenas.push(leg.fri_arena());
        arenas.extend(leg.caps_arena());
    }
    arenas
}

/// A block node over `children` from their PROOF-read shapes, under a
/// weakened binding set — a mutation, for the negatives. Production nodes come
/// from the plan over the children's derived shapes ([`compose_block_tree`]).
pub(super) fn block_node_program_with(
    children: &[&RealChild],
    layout: BlockLayout,
    top: bool,
    checks: BlockBindings,
) -> LfmProgram {
    let shapes: Vec<_> = children.iter().map(|c| child_shape(c)).collect();
    let mut b = production_builder();
    emit_block_node_with(
        &mut b,
        &BlockNodeInputs {
            children: &shapes,
            layout,
            top,
        },
        checks,
    );
    let program = compile(b.finish());
    super::validator::validate(&program).expect("a block node must be admissible");
    program
}

/// The plan's node over `children`, from the shapes their ARTIFACTS give — no
/// child proof — checked against the heights their proofs carry.
pub(super) fn block_node_program(
    plan: &BlockTreePlan,
    children: &[&RealChild],
    top: bool,
) -> LfmProgram {
    let words = plan.child_layout().total();
    let derived: Vec<DerivedChild> = children
        .iter()
        .map(|c| {
            let d = DerivedChild::from_artifacts(&c.artifacts, &c.opts, words)
                .unwrap_or_else(|e| panic!("a child's shapes derive from its artifacts: {e}"));
            let proved: Vec<u32> = c.tables.iter().map(|t| t.shape.log2_trace_length).collect();
            assert_eq!(
                d.log2_trace_lengths(),
                proved,
                "the artifacts' heights are the child proof's trace lengths"
            );
            d
        })
        .collect();
    plan.node_program(&derived, top)
        .unwrap_or_else(|e| panic!("a block node must emit: {e}"))
}

/// A block node's arenas: its children's, in child order.
pub(super) fn block_node_arenas(children: &[&RealChild]) -> Vec<Vec<LfmWord>> {
    children.iter().flat_map(|c| child_arena_words(c)).collect()
}

/// Prove `program` over `arenas` and read the proof back as a child — production
/// must accept it (`real_child_timed`'s assert). Prints one timing line.
pub(super) fn prove_as_child(
    label: &str,
    program: &LfmProgram,
    arenas: &[Vec<LfmWord>],
    opts: &crate::ProofOptions,
) -> RealChild {
    prove_program(label, program, arenas, opts).0
}

/// [`prove_as_child`], keeping the proof — what the tree's final check reads.
pub(super) fn prove_program(
    label: &str,
    program: &LfmProgram,
    arenas: &[Vec<LfmWord>],
    opts: &crate::ProofOptions,
) -> (RealChild, super::proof::LfmProof) {
    let t = std::time::Instant::now();
    let artifacts = super::program_census::build_artifacts_counted(
        program,
        opts,
        crate::hash_pin::BLOCK_HASHER,
    );
    let t_artifacts = t.elapsed().as_secs_f64();
    let t = std::time::Instant::now();
    let proved = super::proof::lfm_prove(program, &artifacts, arenas, opts)
        .unwrap_or_else(|e| panic!("{label} must prove: {e:?}"));
    let t_prove = t.elapsed().as_secs_f64();
    super::per_table_aggregator_tests::print_prove_split(label);
    let t = std::time::Instant::now();
    let (child, t_verify) =
        super::per_table_aggregator_tests::real_child_timed(artifacts, opts.clone(), &proved);
    println!(
        "   {label} TIMING: {} instructions · build_artifacts {t_artifacts:.2}s · prove \
         {t_prove:.2}s · harvest {:.2}s (verify {t_verify:.2})",
        program.instrs.len(),
        t.elapsed().as_secs_f64()
    );
    (child, proved)
}

/// ★ The block tree's FINAL check over the plan the harness holds: the top
/// proof against `expected`, the artifacts of the top program the plan derives
/// with no proof ([`BlockTreePlan::derive_top`]) — never against a program named
/// by whoever produced the proof. The partition is bound only through program
/// identity (a leaf's id commits to its list, each node interns its children's
/// ids), so this is where a tree over any other partition is refused: its top is
/// another program, and its proof fails here. Production's whole verifier,
/// which derives the plan itself, is [`super::block_plan::verify_block_tree`].
pub(super) fn verify_block_top(
    expected: &super::registry::LfmArtifacts,
    top: &super::proof::LfmProof,
    opts: &crate::ProofOptions,
) -> bool {
    super::proof::verify_against_artifacts(expected, &top.proof, &top.public_words, opts)
}

/// Levels above the leaves: fan-in [`super::per_table_aggregator::FAN_IN`],
/// leftovers wrapped in arity-1 nodes (`tree_shape`), the last level's single
/// node the top. A one-leaf block still gets a top node: the bus closes there.
/// Returns the top child, its proof and each level's wall.
pub(super) fn compose_block_tree(
    plan: &BlockTreePlan,
    leaves: Vec<RealChild>,
    opts: &crate::ProofOptions,
    siblings: usize,
) -> (RealChild, super::proof::LfmProof, Vec<f64>) {
    use super::per_table_aggregator::{FAN_IN, Level, tree_shape};
    let mut shape = tree_shape(leaves.len(), FAN_IN);
    if shape.is_empty() {
        shape.push(Level { arities: vec![1] });
    }
    let mut level = leaves;
    let mut top_proof = None;
    let mut walls = Vec::with_capacity(shape.len());
    for (lv, arities) in shape.iter().enumerate() {
        let top = lv + 1 == shape.len();
        let t = std::time::Instant::now();
        let mut starts = Vec::with_capacity(arities.arities.len());
        let mut at = 0usize;
        for a in &arities.arities {
            starts.push(at..at + a);
            at += a;
        }
        assert_eq!(at, level.len(), "the level's arities cover its children");
        let next = super::per_table_aggregator_tests::in_index_order(starts.len(), siblings, |j| {
            let kids: Vec<&RealChild> = level[starts[j].clone()].iter().collect();
            let label = format!(
                "BLOCK L{}N{j} ({} child{}){}",
                lv + 1,
                kids.len(),
                if kids.len() == 1 { "" } else { "ren" },
                if top { " TOP" } else { "" }
            );
            let te = std::time::Instant::now();
            let program = block_node_program(plan, &kids, top);
            println!("   {label}: emitted in {:.2}s", te.elapsed().as_secs_f64());
            super::per_table_aggregator_tests::census_and_panel(&program, &label, FAN_IN);
            let (child, proof) = prove_program(&label, &program, &block_node_arenas(&kids), opts);
            (child, top.then_some(proof))
        });
        let (next, mut proofs): (Vec<RealChild>, Vec<Option<super::proof::LfmProof>>) =
            next.into_iter().unzip();
        if top {
            top_proof = proofs.pop().flatten();
        }
        let wall = t.elapsed().as_secs_f64();
        println!(
            "   BLOCK LEVEL {}: {} node(s) in {wall:.2}s{}",
            lv + 1,
            next.len(),
            if top { " (the top)" } else { "" }
        );
        walls.push(wall);
        level = next;
    }
    assert_eq!(level.len(), 1, "the tree closes to one node");
    (
        level.pop().expect("one"),
        top_proof.expect("the top level keeps its proof"),
        walls,
    )
}

/// The top node's published words, against the host's: the id, the state, the
/// public output.
pub(super) fn assert_top_claims_the_block(top: &RealChild, rb: &RealBlock) {
    let words: Vec<LfmWord> = top.public_words.iter().map(|(_, w)| *w).collect();
    let mut want = super::programs::program_id_words(&rb.plan.attested().program_id()).to_vec();
    want.push(rb.state);
    want.extend(le_halves(&rb.public_output).into_iter().map(base_word));
    assert_eq!(
        words, want,
        "the top node publishes the block's claim: the attestation id, the state and \
         the public output"
    );
    assert!(
        top_claims(&rb.plan, &top.public_words, &rb.public_output),
        "the block verifier's claim check accepts the honest top"
    );
}

// ============================ fixture-scale blocks ========================

/// The small block's options: a real format at blowup 4 (every table's LDE pairs
/// index at least one bit) with two queries, so the leaves stay small.
fn fixture_block_options() -> crate::ProofOptions {
    super::epoch_tests::from_proof_gate_options()
}

/// A small guest proved as one no-epoch block on the host: every table cut at
/// 2^5 rows (`MaxRowsConfig::small`), so most types run several instances.
fn small_block(name: &str, input: &[u8], opts: &crate::ProofOptions) -> (Vec<u8>, crate::VmProof) {
    use executor::vm::execution::Executor;
    let elf_bytes = crate::test_utils::asm_elf_bytes(name);
    let program = executor::elf::Elf::load(&elf_bytes).expect("load the ELF");
    let run = Executor::new(&program, input.to_vec())
        .expect("executor")
        .run()
        .expect("run");
    let mut traces = crate::tables::trace_builder::Traces::from_elf_and_logs(
        &program,
        &run.logs,
        &crate::tables::MaxRowsConfig::small(),
        input,
        #[cfg(feature = "disk-spill")]
        stark::storage_mode::StorageMode::Ram,
    )
    .expect("build the traces");
    let proof = crate::block::prove_block_traces(
        &elf_bytes,
        &program,
        &mut traces,
        opts,
        None,
        stark::residency_mode::ResidencyMode::Retain,
        Vec::new(),
        &mut crate::block::BlockTimes::default(),
    )
    .expect("the small block proves");
    (elf_bytes, proof)
}

/// ★ (b) Block leaves over a real block proof EXECUTE, and publish exactly the
/// host's values: the attestation id, the transcript state after `z, α`, the
/// public output, and shares of the bus that sum to zero over the leaves.
///
/// `poc_rodata_commit` carries an ELF data page (an ELF-derived preprocessed root
/// besides DECODE's) and a public output (a nonzero COMMIT-bus target). Three
/// leaves (a test partition: the plan's own is one leaf at this size), so the
/// shares are split and the target sits in one of them.
///
/// Then, per leaf, the arena census (every declared word is a proof word the
/// shapes prescribe, and nothing more), and the leaf-level refusals: a tampered
/// `L`, a non-canonical output half, and a leaf emitted from any other part
/// count or contribution flag than the AIR's is another program.
#[test]
#[ignore = "proves a VM block (BITWISE is 2^20 rows); box tier"]
fn block_leaves_execute_over_a_real_block_proof() {
    let opts = fixture_block_options();
    let wrap_opts = super::proof::aggregation_wrap_options();
    let (elf_bytes, proof) = small_block("poc_rodata_commit", &[], &opts);
    assert!(
        crate::block::verify_block(&proof, &elf_bytes, &opts).expect("verifies"),
        "the block verifier accepts the small block"
    );
    let (mut rb, verify, replay) = harvest_block(&opts, &elf_bytes, &proof).expect("harvest");
    assert!(!rb.public_output.is_empty(), "a nonzero COMMIT-bus target");
    assert!(
        !rb.plan.attested().pages.is_empty(),
        "an ELF data page is attested"
    );
    println!(
        "BLOCK FIXTURE: the plan's own partition is {} leaf(s)",
        rb.plan.partition().num_leaves()
    );
    let three = rb.partition_over(3);
    rb.plan = rb.plan.with_partition(three);
    let partition = rb.plan.partition().clone();
    println!(
        "BLOCK FIXTURE: {} instances, verify {verify:.2}s · replay {replay:.2}s, leaves {:?}",
        rb.num_instances(),
        partition.leaves()
    );
    let dw = super::proof_arena::words_per_root();
    let mut total = FEE::zero();
    for k in 0..partition.num_leaves() {
        let program = block_leaf_program(&rb, k);
        let arenas = block_leaf_arenas(&rb, &partition, k);
        let exec = execute(&program, &arenas, &crate::hash_pin::BLOCK_HASHER)
            .unwrap_or_else(|e| panic!("leaf {k} must execute: {e:?}"));
        let got: Vec<LfmWord> = exec.public_words.iter().map(|(_, w)| *w).collect();
        assert_eq!(
            got,
            rb.expected_leaf_publics(&partition, k, k == rb.plan.carrier()),
            "leaf {k} publishes the host's id, state, output and share"
        );
        total += word_as_ext(got.last().expect("a share")).expect("an ext share");

        // ---- the census: the schema is the proof's words, counted from the
        // proof's own blocks, and not one more — a surplus word is where an
        // unbound second copy of a value hides.
        let declared: usize = program.arena_schema.lens.iter().map(|l| *l as usize).sum();
        let mut want = rb.public_output.len().div_ceil(4) + dw * rb.num_instances();
        for &i in partition.leaf(k) {
            let (h, leg) = (&rb.tables[i], &rb.legs[i]);
            want += dw * usize::from(h.aux_root.is_some())
                + usize::from(h.contribution.is_some())
                + dw
                + h.ood_current.len()
                + h.ood_next.len()
                + h.parts.len()
                + dw * h.fri_roots.len()
                + h.fri_coeffs.len()
                + usize::from(h.nonce.is_some())
                + leg.opening_arena().len()
                + leg.fri_arena().len()
                + leg.caps_arena().iter().map(Vec::len).sum::<usize>();
        }
        assert_eq!(
            declared, want,
            "leaf {k}: the arena schema is exactly the proof's words"
        );

        // ---- a tampered `L`: the fork absorbs it, so every later challenge
        // moves and the leaf does not execute.
        if let Some((i, at)) = contribution_arena(&rb, &partition, k) {
            let mut bad = arenas.clone();
            assert_eq!(
                bad[at],
                vec![ext_word(&rb.tables[i].contribution.expect("an L"))],
                "the census found instance {i}'s L"
            );
            bad[at][0][0] += FE::from(1u64);
            assert!(
                execute(&program, &bad, &crate::hash_pin::BLOCK_HASHER).is_err(),
                "leaf {k}: a tampered L of instance {i} must not execute"
            );
        }
        println!(
            "BLOCK FIXTURE leaf {k}: {} instances, {} instructions, {declared} arena words, \
             executes",
            partition.leaf(k).len(),
            program.instrs.len()
        );
    }
    assert_eq!(total, FEE::zero(), "the leaves' shares close the bus");

    // ---- a non-canonical output half: the statement decomposes each half into
    // its live bytes, so a half at or over 2^32, or a nonzero pad byte, is refused.
    let program = block_leaf_program(&rb, 0);
    let arenas = block_leaf_arenas(&rb, &partition, 0);
    let half0: u64 = arenas[0][0][0].canonical();
    let mut over = arenas.clone();
    over[0][0] = base_word(FE::from(half0 + (1u64 << 32)));
    assert!(
        execute(&program, &over, &crate::hash_pin::BLOCK_HASHER).is_err(),
        "an output half at or over 2^32 must not execute"
    );
    let live = rb.public_output.len() % 4;
    if live != 0 {
        let last = arenas[0].len() - 1;
        let v: u64 = arenas[0][last][0].canonical();
        let mut padded = arenas.clone();
        padded[0][last] = base_word(FE::from(v | (1u64 << (8 * live))));
        assert!(
            execute(&program, &padded, &crate::hash_pin::BLOCK_HASHER).is_err(),
            "a nonzero pad byte in the last output half must not execute"
        );
    } else {
        println!("BLOCK FIXTURE: the output fills its last half; no pad byte to tamper");
    }

    // ---- M1: a leaf emitted from any other part count or contribution flag
    // than the AIR's is another program — so a tree over it has another top,
    // refused at the final check (the fixture tree's partition negatives show
    // that refusal on proved trees).
    let (k, i) = (0..partition.num_leaves())
        .flat_map(|k| partition.leaf(k).iter().map(move |&i| (k, i)))
        .find(|&(_, i)| rb.plan.instance(i).challenge.has_contribution)
        .expect("an instance with a bus contribution");
    let honest =
        super::block_plan::artifacts_of(&block_leaf_program(&rb, k), &wrap_opts).program_id;
    type Mutation = fn(&mut super::block_plan::PlannedInstance);
    let mutations: [(&str, Mutation); 2] = [
        ("an inflated part count", |p| {
            p.challenge.num_parts += 1;
            p.verify.quotient.num_composition_parts += 1;
            p.verify.sub.deep.num_composition_parts += 1;
        }),
        ("a dropped contribution", |p| {
            p.challenge.has_contribution = false
        }),
    ];
    for (what, mutate) in mutations {
        let mut plan = BlockTreePlan::derive(&elf_bytes, &opts, &BlockShape::of_proof(&proof))
            .expect("the plan derives")
            .with_partition(partition.clone());
        mutate(plan.instance_mut(i));
        let program = plan.leaf_program(k).expect("the mutant emits");
        let id = super::block_plan::artifacts_of(&program, &wrap_opts).program_id;
        assert_ne!(
            id, honest,
            "leaf {k} emitted from {what} on instance {i} is another program"
        );
        println!("BLOCK FIXTURE: {what} on instance {i} makes leaf {k} another program");
    }
}

/// Instance and arena index of the first `L` leaf `k`'s arenas carry, in
/// [`block_leaf_arenas`]' order.
fn contribution_arena(
    rb: &RealBlock,
    partition: &BlockPartition,
    k: usize,
) -> Option<(usize, usize)> {
    let mut at = 2;
    for &i in partition.leaf(k) {
        let (h, leg) = (&rb.tables[i], &rb.legs[i]);
        at += usize::from(h.aux_root.is_some());
        if h.contribution.is_some() {
            return Some((i, at));
        }
        at += 6 + usize::from(h.nonce.is_some()) + 2 + usize::from(leg.caps_arena().is_some());
    }
    None
}

/// ★ (b)–(e) at fixture scale, PROVED: three leaves, the nodes above them and the
/// top, every proof accepted by production's LFM verifier, the top claiming the
/// block; then the two refusals that need real child proofs, each with its
/// mutation:
///
/// - a leaf proved over ANOTHER block — the same guest and shape, a private input
///   that differs outside the committed bytes, so the public output is equal and
///   only the transcript state tells the blocks apart — refused at the first
///   node by the state check, and accepted with that check removed;
/// - leaves none of which subtracts the COMMIT-bus target (their shares sum to
///   the target, not zero), refused by the top's bus assert, and accepted with
///   that assert removed.
#[test]
#[ignore = "proves two VM blocks, their leaves and nodes; box tier"]
fn the_block_fixture_tree_proves_and_refuses_another_blocks_leaf() {
    let opts = fixture_block_options();
    let wrap_opts = super::proof::aggregation_wrap_options();
    let input_a: Vec<u8> = (0u8..16).collect();
    let mut input_b = input_a.clone();
    input_b[13] ^= 0xff;
    let (elf_bytes, proof_a) = small_block("test_private_input_xpage", &input_a, &opts);
    let (_, proof_b) = small_block("test_private_input_xpage", &input_b, &opts);
    assert_eq!(
        proof_a.public_output, proof_b.public_output,
        "the same claim"
    );
    assert_eq!(
        crate::statement::table_count_values(&proof_a.table_counts),
        crate::statement::table_count_values(&proof_b.table_counts),
        "the same shape"
    );
    let (mut a, ..) = harvest_block(&opts, &elf_bytes, &proof_a).expect("harvest A");
    let (mut b, ..) = harvest_block(&opts, &elf_bytes, &proof_b).expect("harvest B");
    assert_ne!(a.state, b.state, "the blocks differ in their roots");
    // A test partition of three leaves (the plan's own is one at this size), in
    // both plans: the final check below derives over it.
    let partition = a.partition_over(3);
    assert_eq!(partition, b.partition_over(3), "the same partition");
    a.plan = a.plan.with_partition(partition.clone());
    b.plan = b.plan.with_partition(partition.clone());
    let layout = a.plan.child_layout();

    // Leaves of A, and of B; the same programs, since the shapes are equal. The
    // plan's carrier unless `carriers` names others (the negatives).
    let leaves = |rb: &RealBlock, carriers: &[usize], tag: &str| -> Vec<RealChild> {
        (0..partition.num_leaves())
            .map(|k| {
                let carries = carriers.contains(&k);
                let program = if carriers == [rb.plan.carrier()] {
                    block_leaf_program(rb, k)
                } else {
                    block_leaf_program_over(rb, &partition, k, carries)
                };
                let child = prove_as_child(
                    &format!("{tag} leaf {k}"),
                    &program,
                    &block_leaf_arenas(rb, &partition, k),
                    &wrap_opts,
                );
                let words: Vec<LfmWord> = child.public_words.iter().map(|(_, w)| *w).collect();
                assert_eq!(words, rb.expected_leaf_publics(&partition, k, carries));
                child
            })
            .collect()
    };
    let leaves_a = leaves(&a, &[0], "A");
    let leaves_b = leaves(&b, &[0], "B");
    assert_eq!(
        leaves_a[1].artifacts.program_id, leaves_b[1].artifacts.program_id,
        "a leaf's program is a function of the shape, so B's leaf 1 is A's program"
    );

    // ---- another block's leaf, beside A's leaf 0, at the first node.
    let mixed = [&leaves_a[0], &leaves_b[1]];
    let node = |checks: BlockBindings| {
        let program = block_node_program_with(&mixed, layout, false, checks);
        execute(
            &program,
            &block_node_arenas(&mixed),
            &crate::hash_pin::BLOCK_HASHER,
        )
    };
    assert!(
        node(BlockBindings::ALL).is_err(),
        "a node over two blocks' leaves must not execute"
    );
    assert!(
        node(BlockBindings::without("state")).is_ok(),
        "with the state check removed it executes: the state check is what refuses it"
    );

    // ---- no leaf, or two, carry the target: the shares sum to E or to −E, and
    // the top refuses.
    for (what, carriers) in [("no leaf", &[][..]), ("two leaves", &[0, 1][..])] {
        let wrong_leaves = leaves(&a, carriers, &format!("A carried by {what}"));
        let wrong: Vec<&RealChild> = wrong_leaves.iter().collect();
        let top_over = |checks: BlockBindings| {
            let program = block_node_program_with(&wrong, layout, true, checks);
            execute(
                &program,
                &block_node_arenas(&wrong),
                &crate::hash_pin::BLOCK_HASHER,
            )
        };
        assert!(
            top_over(BlockBindings::ALL).is_err(),
            "a top over shares {what} carries must not execute: they do not close the bus"
        );
        assert!(
            top_over(BlockBindings::without("bus")).is_ok(),
            "{what} carrying: with the bus assert removed it executes, so the bus assert \
             is what refuses it"
        );
    }

    // ---- the honest tree over A, and the final check against the top program
    // the verifier derives from the partition.
    let (top, top_proof, walls) = compose_block_tree(&a.plan, leaves_a, &wrap_opts, 1);
    assert_top_claims_the_block(&top, &a);
    let expected = a.plan.derive_top(&wrap_opts).expect("the top derives");
    assert_eq!(
        expected.program_id, top.artifacts.program_id,
        "the plan derives, with no proof, the top program the harness proved"
    );
    assert!(verify_block_top(&expected, &top_proof, &wrap_opts));
    println!("BLOCK FIXTURE TREE: levels {walls:?}, the top claims block A");

    // ---- a tree over ANOTHER partition of the same block: one instance moved from
    // leaf 1 to leaf 2. Every proof in it is honest and its top claims the same
    // block — only its program identity differs, and the final check refuses it.
    let mut lists = partition.leaves().to_vec();
    let moved = lists[1].pop().expect("leaf 1 has instances");
    assert!(!lists[1].is_empty(), "leaf 1 keeps an instance");
    lists[2].push(moved);
    let other = BlockPartition::new(lists, a.num_instances()).expect("still a partition");
    let other_leaves: Vec<RealChild> = (0..other.num_leaves())
        .map(|k| {
            prove_as_child(
                &format!("A other-partition leaf {k}"),
                &block_leaf_program_over(&a, &other, k, k == 0),
                &block_leaf_arenas(&a, &other, k),
                &wrap_opts,
            )
        })
        .collect();
    let (other_top, other_proof, _) = compose_block_tree(&a.plan, other_leaves, &wrap_opts, 1);
    assert_eq!(
        other_top.public_words, top.public_words,
        "the other partition's top claims the same block"
    );
    assert_ne!(other_top.artifacts.program_id, top.artifacts.program_id);
    assert!(
        !verify_block_top(&expected, &other_proof, &wrap_opts),
        "a tree over another partition must be refused at the final check"
    );
    assert!(
        verify_block_top(&other_top.artifacts, &other_proof, &wrap_opts),
        "checked against the program its own prover built, it verifies: the final check's \
         program, derived by the verifier from the partition constant, is what refuses it"
    );
    println!("BLOCK FIXTURE TREE: another partition's tree is refused at the final check");

    // ---- the dangerous trees: one that SKIPS an instance (its constraints are
    // never verified) and one that verifies an instance TWICE. `BlockPartition::new`
    // refuses both list sets, but nothing binds a prover to it. The instance has a
    // zero bus contribution, so leaving it out or counting it twice keeps the bus
    // closed: every in-circuit check passes, the tree proves, and its top claims
    // the same block. Only the final check's program, derived from the verifier's
    // own partition, refuses it.
    let n = a.num_instances();
    let lists = partition.leaves().to_vec();
    let (k_leaf, skipped) = lists
        .iter()
        .enumerate()
        .filter(|(_, l)| l.len() > 1)
        .flat_map(|(j, l)| l.iter().map(move |&i| (j, i)))
        .find(|&(_, i)| a.tables[i].contribution.is_none_or(|c| c == FEE::zero()))
        .expect(
            "the skip/duplicate negatives need an instance with a zero bus contribution in a \
             leaf of two or more",
        );
    println!(
        "BLOCK FIXTURE TREE: instance {skipped} ({}) of leaf {k_leaf} has a zero bus contribution",
        a.names()[skipped]
    );
    let mut skip = lists.clone();
    skip[k_leaf].retain(|&i| i != skipped);
    let mut twice = lists.clone();
    let other_leaf = (k_leaf + 1) % twice.len();
    twice[other_leaf].push(skipped);
    twice[other_leaf].sort_unstable();
    for (what, lists) in [("skips", skip), ("verifies twice", twice)] {
        assert!(
            BlockPartition::new(lists.clone(), n).is_err(),
            "the emitter's partition refuses a tree that {what} an instance"
        );
        let bad = BlockPartition::unvalidated(lists, n);
        let bad_leaves: Vec<RealChild> = (0..bad.num_leaves())
            .map(|k| {
                prove_as_child(
                    &format!("A tree that {what} instance {skipped}, leaf {k}"),
                    &block_leaf_program_over(&a, &bad, k, k == 0),
                    &block_leaf_arenas(&a, &bad, k),
                    &wrap_opts,
                )
            })
            .collect();
        let (bad_top, bad_proof, _) = compose_block_tree(&a.plan, bad_leaves, &wrap_opts, 1);
        assert_eq!(
            bad_top.public_words, top.public_words,
            "a tree that {what} an instance claims the same block"
        );
        assert_ne!(bad_top.artifacts.program_id, top.artifacts.program_id);
        assert!(
            !verify_block_top(&expected, &bad_proof, &wrap_opts),
            "a tree that {what} an instance must be refused at the final check"
        );
        assert!(
            verify_block_top(&bad_top.artifacts, &bad_proof, &wrap_opts),
            "checked against the program its own prover built, a tree that {what} an \
             instance verifies: the derived top program is what refuses it"
        );
        println!(
            "BLOCK FIXTURE TREE: a tree that {what} instance {skipped} is refused at the final check"
        );
    }
}

/// ★ M1 end to end at fixture scale: the block VERIFIER
/// ([`super::block_plan::verify_block_tree`]) derives the plan and the top
/// program from the trusted ELF, the options and the claimed shape — no proof
/// read — and accepts the honest tree's top proof over the plan's OWN partition
/// and carrier. It refuses the same top proof under another ELF, another
/// claimed output, and a shape that lies about one trace length (each derives
/// another top or another claim).
#[test]
#[ignore = "proves a VM block and its tree; box tier"]
fn the_block_verifier_derives_the_tree_and_accepts_only_its_top() {
    let opts = fixture_block_options();
    let wrap_opts = super::proof::aggregation_wrap_options();
    let (elf_bytes, proof) = small_block("poc_rodata_commit", &[], &opts);
    let shape = BlockShape::of_proof(&proof);
    let (rb, ..) = harvest_block(&opts, &elf_bytes, &proof).expect("harvest");
    let partition = rb.plan.partition().clone();
    println!(
        "BLOCK VERIFIER FIXTURE: the plan's partition is {} leaf(s), {} instances",
        partition.num_leaves(),
        rb.num_instances()
    );
    let leaves: Vec<RealChild> = (0..partition.num_leaves())
        .map(|k| {
            prove_as_child(
                &format!("plan leaf {k}"),
                &block_leaf_program(&rb, k),
                &block_leaf_arenas(&rb, &partition, k),
                &wrap_opts,
            )
        })
        .collect();
    let (top, top_proof, _) = compose_block_tree(&rb.plan, leaves, &wrap_opts, 1);
    assert_top_claims_the_block(&top, &rb);

    let verify = |elf: &[u8], shape: &BlockShape, output: &[u8]| {
        super::block_plan::verify_block_tree(elf, &opts, &wrap_opts, shape, output, &top_proof)
    };
    verify(&elf_bytes, &shape, &proof.public_output).expect("the verifier accepts the block");
    let (_, derived) = super::block_plan::derive_block_top(&elf_bytes, &opts, &wrap_opts, &shape)
        .expect("derives");
    assert_eq!(
        derived.program_id, top.artifacts.program_id,
        "the verifier's top is the program the harness proved"
    );

    let other_elf = crate::test_utils::asm_elf_bytes("test_private_input_xpage");
    let other = verify(&other_elf, &shape, &proof.public_output);
    assert!(
        other.is_err(),
        "another ELF's verifier must refuse the block"
    );
    let mut output = proof.public_output.clone();
    output[0] ^= 1;
    assert!(
        verify(&elf_bytes, &shape, &output).is_err(),
        "another claimed output must be refused"
    );
    let mut lie = shape.clone();
    let i = (0..lie.trace_lengths.len())
        .find(|&i| !rb.plan.instance(i).name.starts_with("PAGE") && lie.trace_lengths[i] < 1 << 10)
        .expect("a small table");
    lie.trace_lengths[i] *= 2;
    assert!(
        verify(&elf_bytes, &lie, &proof.public_output).is_err(),
        "a shape that lies about instance {i}'s trace length must be refused"
    );
    println!(
        "BLOCK VERIFIER FIXTURE: accepts the plan's tree; refuses another ELF ({}), another \
         output, a trace-length lie",
        other.err().unwrap_or_default()
    );
}

// ============================ production scale ============================

/// D-NOEPOCH §12.2's partition of block 25368371's 137 instances, as the
/// designer's script printed it: the rule over `legmodel.py`'s costs, which
/// differ from the closed form the plan prices with. Reported beside the plan's
/// partition, never substituted for it.
const D12_LEAVES: [&[usize]; 8] = [
    &[
        0, 1, 2, 3, 4, 5, 7, 13, 49, 61, 69, 77, 85, 93, 101, 109, 117, 125, 133,
    ],
    &[
        8, 12, 37, 45, 53, 62, 70, 78, 86, 94, 103, 111, 119, 127, 136,
    ],
    &[
        9, 11, 35, 42, 50, 59, 67, 75, 83, 91, 99, 105, 113, 121, 129,
    ],
    &[
        6, 10, 34, 41, 48, 54, 60, 68, 76, 84, 92, 100, 108, 116, 124, 132,
    ],
    &[
        14, 18, 22, 26, 31, 38, 46, 55, 63, 71, 79, 87, 95, 106, 114, 122, 130, 134,
    ],
    &[
        15, 19, 23, 27, 32, 39, 43, 51, 58, 66, 74, 82, 90, 98, 107, 115, 123, 131,
    ],
    &[
        16, 20, 24, 28, 33, 40, 47, 56, 57, 65, 73, 81, 89, 97, 104, 112, 120, 128,
    ],
    &[
        17, 21, 25, 29, 30, 36, 44, 52, 64, 72, 80, 88, 96, 102, 110, 118, 126, 135,
    ],
];

/// `NOEPOCH_LEAVES`: the leaf count; unset is the rule's `⌈Σ cost / 279 000⌉`.
fn leaves_knob() -> Option<usize> {
    std::env::var("NOEPOCH_LEAVES")
        .ok()
        .filter(|v| !v.is_empty())
        .map(|v| {
            v.parse()
                .ok()
                .filter(|k: &usize| *k >= 1)
                .unwrap_or_else(|| panic!("NOEPOCH_LEAVES must be a positive integer, got `{v}`"))
        })
}

/// ★★★ THE WHOLE NO-EPOCH BLOCK, base to top node: `block::prove_block`, the
/// harvest, the leaves (level 0), the interior and the top — the block's
/// recursion, and the number the epoch/no-epoch decision is taken on.
///
/// Its `★★★ WHOLE RUN` line is measured as the epoch tree's
/// ([`super::per_table_aggregator_tests::the_production_tree_composes_to_a_root`]):
/// from before the base to the last proof, the harness's verifies inside. That
/// tree's number stops at its interior (its block-artifact root is a separate
/// arm); this one's top node is its last proof, and it closes the block's bus.
#[test]
#[ignore = "box tier, production scale: the whole no-epoch block and its recursion (NOEPOCH_ELF, NOEPOCH_INPUT, --features cuda)"]
fn the_block_tree_composes_to_a_top_node() {
    use super::per_table_aggregator_tests::{
        HostSampler, cgroup_limit_gib, in_index_order, tree_siblings, tree_siblings_l0,
    };
    use std::time::Instant;

    // A refused harvest names its table and its check: the verifier's `error!` lines.
    let _ = env_logger::builder().is_test(true).try_init();

    if !cfg!(feature = "cuda") {
        panic!("the production block tree requires `--features cuda`");
    }
    let read = |var: &str| -> Vec<u8> {
        let p = std::env::var(var).unwrap_or_else(|_| panic!("{var} must name a file"));
        std::fs::read(&p).unwrap_or_else(|e| panic!("{var} {p}: {e}"))
    };
    let elf_bytes = read("NOEPOCH_ELF");
    let input = read("NOEPOCH_INPUT");
    let inner = super::proof::block_base_options();
    let wrap_opts = super::proof::aggregation_wrap_options();
    let ceiling = cgroup_limit_gib();
    let pct = |g: f64| match &ceiling {
        Ok(c) => format!(" ({:.1}% of {c:.2})", 100.0 * g / c),
        Err(_) => String::new(),
    };
    println!(
        "★★★ NO-EPOCH BLOCK TREE (base + leaves + interior + top)\n   \
         {} input bytes · inner blowup {} / {} q · wrap blowup {} / {} q · cgroup {}",
        input.len(),
        inner.blowup_factor,
        inner.fri_number_of_queries,
        wrap_opts.blowup_factor,
        wrap_opts.fri_number_of_queries,
        match &ceiling {
            Ok(g) => format!("{g:.2} GiB"),
            Err(why) => format!("UNKNOWN — {why}"),
        },
    );

    let whole = HostSampler::start();
    let t_all = Instant::now();

    // ---- the base.
    let base_sampler = HostSampler::start();
    let t = Instant::now();
    let (proof, times) =
        crate::block::prove_block(&elf_bytes, &input, &inner).expect("the block must prove");
    let base = t.elapsed().as_secs_f64();
    let (base_peak, _) = base_sampler.stop();
    println!(
        "   base: {} sub-proofs in {base:.2}s (execute {:.2} · build {:.2} · setup {:.2} · \
         prove {:.2}) · host peak {base_peak:.3} GiB{}",
        proof.proof.proofs.len(),
        times.execute,
        times.build,
        times.setup,
        times.prove,
        pct(base_peak)
    );

    // ---- the harvest. Production's verify of the base is a harness assert, not
    // work a driver does, so by default it runs on a helper thread beside level 0
    // and is joined before anything is reported: a refused block still fails the
    // run, it only stops delaying the leaves (the epoch tree's per-epoch verifies
    // likewise run beside its other wraps). `NOEPOCH_HARVEST_VERIFY=inline`
    // verifies first, as before.
    let inline_verify = std::env::var("NOEPOCH_HARVEST_VERIFY").is_ok_and(|v| v == "inline");
    let t = Instant::now();
    let shape = BlockShape::of_proof(&proof);
    let (mut rb, verify, replay, beside) = if inline_verify {
        let (rb, verify, replay) = harvest_block(&inner, &elf_bytes, &proof).expect("harvest");
        drop(proof);
        (rb, Some(verify), replay, None)
    } else {
        let proof = std::sync::Arc::new(proof);
        let beside = {
            let (proof, opts, elf) = (proof.clone(), inner.clone(), elf_bytes.clone());
            std::thread::spawn(move || {
                harvest_block(&opts, &elf, &proof).map(|(rb, verify, _)| (rb.state, verify))
            })
        };
        let (rb, _, replay) =
            harvest_block_with(&inner, &elf_bytes, &proof, false).expect("harvest");
        drop(proof);
        (rb, None, replay, Some(beside))
    };
    let mut harvest = t.elapsed().as_secs_f64();
    println!(
        "   harvest: {harvest:.2}s (production verify {}, harness-only · replay {replay:.2}s) · \
         {} instances · public output {} bytes",
        match verify {
            Some(v) => format!("{v:.2}s inline"),
            None => "beside level 0".to_string(),
        },
        rb.num_instances(),
        rb.public_output.len()
    );

    // ---- the partition: the plan's — the rule over the closed-form costs, a
    // pure function of the block's shape. `NOEPOCH_LEAVES` forces another leaf
    // count: another tree, which the final check then derives over.
    let forced = leaves_knob();
    if let Some(k) = forced {
        let p = rb.partition_over(k);
        rb.plan = rb.plan.with_partition(p);
    }
    let partition = rb.plan.partition().clone();
    let names = rb.names();
    let d12_block = names.len() == 137 && names[42] == "MEMW[0]" && names[100] == "MEMW_R[0]";
    println!(
        "   BLOCK PARTITION SOURCE: {} · D-NOEPOCH §12.2's lists (legmodel.py costs): {}",
        if forced.is_some() {
            "NOEPOCH_LEAVES (forced: NOT the plan's tree)"
        } else {
            "the plan (the rule)"
        },
        if !d12_block {
            "n/a (not the 137-instance block)"
        } else if partition
            .leaves()
            .iter()
            .map(Vec::as_slice)
            .eq(D12_LEAVES.iter().copied())
        {
            "equal"
        } else {
            "differ (reported, not used)"
        }
    );
    let costs = rb.plan.costs();
    let k = partition.num_leaves();
    for (j, leaf) in partition.leaves().iter().enumerate() {
        let perms: usize = leaf.iter().map(|&i| costs[i]).sum();
        println!(
            "   BLOCK LEAF {j}: {} instances · {perms} perms (closed form) · idx {leaf:?}",
            leaf.len()
        );
    }
    println!(
        "   BLOCK PARTITION: {k} leaves, Σ {} perms, heaviest {} (cap {})",
        costs.iter().sum::<usize>(),
        partition
            .leaves()
            .iter()
            .map(|l| l.iter().map(|&i| costs[i]).sum::<usize>())
            .max()
            .unwrap_or(0),
        super::block_plan::LEAF_PERMS_CAP
    );

    // ---- level 0: the leaves.
    let l0 = tree_siblings_l0().min(k);
    println!("   ★ LEVEL-0 CONCURRENCY: {l0} leaf proof(s) at once (LFM_TREE_SIBLINGS_L0)");
    super::device_permit::arm(l0);
    let l0_sampler = HostSampler::start();
    let t = Instant::now();
    let leaves = in_index_order(k, l0, |j| {
        let label = format!("BLOCK L0 leaf {j}");
        let te = Instant::now();
        let program = block_leaf_program(&rb, j);
        let arenas = block_leaf_arenas(&rb, &partition, j);
        println!(
            "   {label}: emitted + arenas in {:.2}s",
            te.elapsed().as_secs_f64()
        );
        super::per_table_aggregator_tests::census_and_panel(
            &program,
            &label,
            super::per_table_aggregator::FAN_IN,
        );
        let child = prove_as_child(&label, &program, &arenas, &wrap_opts);
        let words: Vec<LfmWord> = child.public_words.iter().map(|(_, w)| *w).collect();
        assert_eq!(
            words,
            rb.expected_leaf_publics(&partition, j, j == rb.plan.carrier()),
            "{label} publishes the host's id, state, output and share"
        );
        child
    });
    let level0 = t.elapsed().as_secs_f64();
    let (l0_peak, _) = l0_sampler.stop();
    println!(
        "   BLOCK LEVEL 0: {k} leaves in {level0:.2}s · host peak {l0_peak:.3} GiB{}",
        pct(l0_peak)
    );

    // ---- the interior and the top.
    let siblings = tree_siblings();
    println!("   ★ SIBLING CONCURRENCY: {siblings} node proof(s) at once (LFM_TREE_SIBLINGS)");
    super::device_permit::arm(siblings);
    let t = Instant::now();
    let (top, top_proof, walls) = compose_block_tree(&rb.plan, leaves, &wrap_opts, siblings);
    super::device_permit::arm(1);
    let interior = t.elapsed().as_secs_f64();
    // The verify beside level 0 must have accepted the block, over the same
    // reconstruction the leaves read (one state word), before anything counts.
    // What the join waits is on the critical path, so it is the harvest's.
    if let Some(beside) = beside {
        let t = Instant::now();
        let (state, verify) = beside
            .join()
            .expect("the harvest verify beside level 0 panicked")
            .expect("harvest (verified beside level 0)");
        assert_eq!(
            state, rb.state,
            "the verified harvest and the leaves' harvest read one transcript"
        );
        let waited = t.elapsed().as_secs_f64();
        harvest += waited;
        println!(
            "   harvest verify beside level 0: production verify {verify:.2}s, harness-only · \
             joined after the top, waited {waited:.2}s (counted in the harvest)"
        );
    }
    assert_top_claims_the_block(&top, &rb);
    let t = Instant::now();
    assert!(
        verify_block_top(&top.artifacts, &top_proof, &wrap_opts),
        "the top proof must verify against the top program the harness emitted"
    );
    println!(
        "   BLOCK FINAL CHECK (harness): the top verifies against its emitted program \
         ({:.2}s, program id {})",
        t.elapsed().as_secs_f64(),
        top.artifacts
            .program_id
            .iter()
            .map(|b| format!("{b:02x}"))
            .collect::<String>()
    );

    let total = t_all.elapsed().as_secs_f64();
    let (peak, at) = whole.stop();
    println!(
        "★★★ NO-EPOCH BLOCK: base {base:.2}s · harvest {harvest:.2}s · level 0 {level0:.2}s \
         ({k} leaves) · interior {interior:.2}s (levels {}) · recursion {:.2}s · whole {total:.2}s",
        walls
            .iter()
            .map(|w| format!("{w:.2}"))
            .collect::<Vec<_>>()
            .join(" + "),
        harvest + level0 + interior
    );
    println!("★★★ WHOLE RUN: host peak {peak:.3} GiB at t={at:.1}, {total:.1}s total");

    // ---- the block VERIFIER, outside the whole run (a consumer's work, not the
    // prover's): the plan and the top program derived from the ELF, the options
    // and the claimed shape with no proof read, the top proof verified against
    // that program, its words against the ELF's id and the output.
    let t = Instant::now();
    let derived = match forced {
        None => super::block_plan::derive_block_top(&elf_bytes, &inner, &wrap_opts, &shape)
            .map(|(_, top)| top),
        Some(_) => rb.plan.derive_top(&wrap_opts),
    }
    .expect("the verifier derives the top program");
    let derive_secs = t.elapsed().as_secs_f64();
    assert_eq!(
        derived.program_id, top.artifacts.program_id,
        "the verifier derives, with no proof, the top program the harness proved"
    );
    assert!(
        verify_block_top(&derived, &top_proof, &wrap_opts),
        "the top proof verifies against the derived top program"
    );
    assert!(
        top_claims(&rb.plan, &top_proof.public_words, &rb.public_output),
        "the top proof claims the trusted ELF's id and the block's output"
    );
    println!(
        "   BLOCK VERIFIER: {} plan + top program derived in {derive_secs:.2}s (no proof read) · \
         the top proof verifies against it and claims the block · {:.2}s in all",
        if forced.is_some() {
            "forced-partition"
        } else {
            "production"
        },
        t.elapsed().as_secs_f64()
    );
}
