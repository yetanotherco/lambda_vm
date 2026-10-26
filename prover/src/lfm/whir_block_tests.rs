//! The no-epoch WHIR block's recursion ([`super::whir_block`]): the group
//! partition; the leaves over small block proofs, closing the bus between them;
//! a refusal per check a leaf or a node adds, each shown load-bearing by its
//! mutation; the tree to the top the plan derives; and the real block (box).

use crate::block_whir::{
    self, BlockFormat, BlockOptions, BlockWhirProof, prove_block_whir, verify_block_whir,
};
use crate::hash_pin::{BlockFamily, ProofFamily};
use crate::tables::MaxRowsConfig;
use crate::tables::types::{FE, FEE};
use crate::test_utils::E as FEE_FIELD;
use crate::test_utils::asm_elf_bytes;
use crate::zf_format::ZfFormat;
use multilinear::whir_chain::{ArgueFormat, StackVars};
use stark::proof::options::ProofOptions;

use super::block_node::{
    BlockBindings, BlockLayout, BlockNodeInputs, bind_and_publish_with, emit_block_node_with,
};
use super::builder::LfmBuilder;
use super::compiler::{LfmProgram, compile};
use super::executor::execute;
use super::per_table_aggregator::{DerivedChild, LegCells, hint_public_words, publics_arena};
use super::per_table_aggregator_tests::{RealChild, child_arena_words};
use super::proof::{LfmProof, aggregation_wrap_options, lfm_prove_under};
use super::whir_block::{
    BLOCK_FAN_IN, BlockPartition, LeafChecks, WhirBlockPlan, artifacts_of, block_leaf_arena,
    emit_share, group_arena_words, id_words, leaf_arena, leaf_partition, leaf_program_with,
    out_halves, partition_groups, verify_block_tree_under,
};
use super::whir_block_tree::{
    FailEmpty, FailUnpublished, GroupMsg, ProgramSlot, Published, TreeAt, TreeBudget, build_levels,
    in_arrival_order, node_flow, prove_dataflow, publish_each, top_streams, tree_order, wait_all,
};
use super::word::{LfmWord, base_word, word_as_ext};

/// A stack of 2^10 and two polynomials a group: a small program spans several
/// groups, which is the block's shape at a size a test can prove.
fn small_format() -> BlockFormat {
    let mut zf = ZfFormat::DEFAULT;
    zf.whir_stack = StackVars::new(10).expect("a valid stack");
    BlockFormat {
        zf,
        group_polys: 2,
        max_groups: block_whir::BLOCK_MAX_GROUPS,
        prepared: true,
        argue: ArgueFormat::PerTable,
        group: multilinear::whir_chain::GroupFold::PerPoly,
        leaf_scale: crate::lfm::whir_block::LeafScale::Fixed,
        leaf_query: crate::lfm::whir_chain::LeafQuery::V1,
    }
}

/// `format` at blowup 4, for the tests whose blocks the leaf planner holds to a
/// leaf's rows. The test format's 2^10 stack makes a lone wide table (BITWISE's
/// 21 columns of 2^20 rows are 21 polynomials on their own), whose leaf rows at
/// blowup 2 pass a leaf's room: a shape production's 2^27 stack cannot form.
/// These tests check the leaves' and the tree's mechanics, which the rate does
/// not change.
fn at_blowup_4(format: BlockFormat) -> BlockFormat {
    BlockFormat {
        zf: ZfFormat {
            whir_blowup: 4,
            ..format.zf
        },
        ..format
    }
}

/// The same with three polynomials a group: the same program packs into other
/// groups, so its statement — and every leaf's state and id — differ.
fn other_format() -> BlockFormat {
    BlockFormat {
        group_polys: 3,
        ..small_format()
    }
}

fn small_block(name: &str, format: &BlockFormat) -> (Vec<u8>, BlockWhirProof) {
    small_block_at(name, format, block_whir::BLOCK_ECDAS_ROWS_LOG2)
}

/// [`small_block`] with ECDAS cut at 2^`ecdas_rows_log2` rows.
fn small_block_at(
    name: &str,
    format: &BlockFormat,
    ecdas_rows_log2: usize,
) -> (Vec<u8>, BlockWhirProof) {
    small_block_cut(name, format, |o| o.ecdas_rows_log2 = ecdas_rows_log2)
}

/// [`small_block`] with its options changed by `cut` (the chunked tables'
/// heights).
fn small_block_cut(
    name: &str,
    format: &BlockFormat,
    cut: impl FnOnce(&mut BlockOptions),
) -> (Vec<u8>, BlockWhirProof) {
    let elf = asm_elf_bytes(name);
    let opts = ProofOptions::default_test_options();
    let mut options = BlockOptions {
        max_rows: MaxRowsConfig::small(),
        keccak_rnd_rows_log2: 3,
        ecdas_rows_log2: block_whir::BLOCK_ECDAS_ROWS_LOG2,
        keccak_rows_log2: block_whir::BLOCK_KECCAK_ROWS_LOG2,
        ecsm_rows_log2: block_whir::BLOCK_ECSM_ROWS_LOG2,
        copy_rows_log2: block_whir::BLOCK_COPY_ROWS_LOG2,
        drop_levels: multilinear::whir_commit::TreeDrop::uniform(3),
        window_log2: None,
        stream_keccak_rnd: false,
        stream_memw_lt: false,
        drop_streamed_ops: false,
        layout_workers: 0,
        layout_ahead: Some(2),
        pack_rest_as_laid_out: false,
        // Every table packed on a card, so a box run walks the narrow path at a
        // test's size.
        narrow: stark::multilinear_block::Narrowing::Card { min_cells: 0 },
        upload_ahead: true,
        memlog: false,
        // The rest laid out in waves of 1 MiB: several at a test's size.
        finish_keccak_rnd_chunks: true,
        finish_cuts: true,
        finish_stream: true,
        rest_layout_bytes: Some(1 << 20),
        rest_build_bytes: None,
        pack_finished: true,
        gpack: true,
        spill: crate::block_whir::BlockSpillPolicy::Off,
        regen: None,
        phase_b_overlap: stark::multilinear_block::PhaseBOverlap::Off,
    };
    cut(&mut options);
    let proof = prove_block_whir(&elf, &[], &opts, format, &options)
        .expect("the block proves")
        .0;
    assert!(
        verify_block_whir(&proof, &elf, &opts, format).expect("verify runs"),
        "the host verifier accepts the block"
    );
    (elf, proof)
}

fn plan_of(
    elf: &[u8],
    proof: &BlockWhirProof,
    format: &BlockFormat,
    leaves: Option<usize>,
) -> WhirBlockPlan {
    WhirBlockPlan::derive(
        elf,
        &ProofOptions::default_test_options(),
        format,
        proof.statement(),
        leaves,
    )
    .expect("the plan derives")
}

/// Leaf `k` executed over the proof's arena: its published words, or why it
/// did not execute.
fn run_leaf(
    plan: &WhirBlockPlan,
    proof: &BlockWhirProof,
    k: usize,
    checks: LeafChecks,
) -> Result<Vec<LfmWord>, String> {
    let program = leaf_program_with(plan, k, checks)?;
    let arenas = block_leaf_arena(plan, proof, k)?;
    assert_eq!(
        program.arena_schema.lens,
        vec![arenas[0].len() as u32],
        "the leaf hints exactly the words its arena holds"
    );
    execute(
        &program,
        &arenas,
        &program.hasher(<BlockFamily as ProofFamily>::SOCKET),
    )
    .map(|e| e.public_words.iter().map(|(_, w)| *w).collect())
    .map_err(|e| format!("{e:?}"))
}

fn leaves_words(plan: &WhirBlockPlan, proof: &BlockWhirProof) -> Vec<Vec<LfmWord>> {
    (0..plan.partition().num_leaves())
        .map(|k| run_leaf(plan, proof, k, LeafChecks::ALL).expect("an honest leaf executes"))
        .collect()
}

fn ext_of(word: &LfmWord) -> FEE {
    word_as_ext(word).expect("an extension word")
}

/// ★ What phase B hands over as each group's opening ends is the finished
/// proof's share of that group: built from the observer's groups alone, every
/// leaf's arena is the one the proof gives, under both argue formats, for a
/// streamed block (the production path).
///
/// At blowup 4 ([`at_blowup_4`]): the test format's small stack makes a lone
/// wide table that production cannot form, and at blowup 2 its leaf rows
/// pass a leaf's room. The lean arm stays at blowup 2, the shipped rate: the
/// lean leaf holds the format's widest groups at 0.821 (BITWISE) and 0.802
/// (CPU) of a leaf (`leaf_room_small_format_census`), so a rise past a leaf
/// refuses here first.
#[test]
#[ignore = "proves small blocks; box tier"]
fn the_groups_phase_b_hands_over_build_every_leafs_arena() {
    for (format, name) in [
        (at_blowup_4(small_format()), "all_instructions_64"),
        (at_blowup_4(small_batched()), "all_instructions_64"),
        (at_blowup_4(small_batched()), "test_keccak"),
        // At blowup 2, the shipped rate: the lean leaf holds this format's
        // widest groups (0.82 of a leaf), where V1's do not.
        (lean_format(small_format()), "all_instructions_64"),
    ] {
        let elf = asm_elf_bytes(name);
        let opts = ProofOptions::default_test_options();
        let mut options = BlockOptions::production();
        options.max_rows = MaxRowsConfig::small();
        options.keccak_rnd_rows_log2 = 3;
        options.drop_levels = multilinear::whir_commit::TreeDrop::uniform(3);
        options.window_log2 = Some(4);
        options.narrow = stark::multilinear_block::Narrowing::Card { min_cells: 0 };
        let seen = std::sync::Mutex::new(Vec::<GroupMsg>::new());
        let on_group = |opened: stark::multilinear_block::GroupOpened<'_, _, _>| {
            seen.lock().expect("lock").push(GroupMsg {
                group: opened.group,
                roots: opened.roots.to_vec(),
                commit_ood: opened.commit_ood.to_vec(),
                tables: opened.tables.to_vec(),
                argue: opened.argue.cloned(),
                opening: opened.opening.clone(),
                prepared: opened.prepared.cloned(),
                at: 0.0,
            });
        };
        let (proof, _) = block_whir::prove_block_whir_observed_groups(
            &elf,
            &[],
            &opts,
            &format,
            &options,
            &|_, _| {},
            &on_group,
        )
        .expect("the block proves");
        let seen = seen.into_inner().expect("lock");
        assert_eq!(
            seen.len(),
            proof.groups.len(),
            "{name}: one hand-over a group"
        );
        assert!(
            seen.iter().enumerate().all(|(g, m)| m.group == g),
            "{name}: the groups in order"
        );
        let plan = plan_of(&elf, &proof, &format, None);
        let words: Vec<Vec<LfmWord>> = seen
            .iter()
            .map(|m| {
                group_arena_words(
                    &plan,
                    m.group,
                    &m.tables,
                    m.argue.as_ref(),
                    &m.opening,
                    m.prepared.as_ref(),
                )
                .expect("the group's words")
            })
            .collect();
        for k in 0..plan.partition().num_leaves() {
            let streamed = leaf_arena(
                &seen[0].roots,
                &seen[0].commit_ood,
                plan.partition()
                    .leaf(k)
                    .iter()
                    .map(|&g| words[g].clone())
                    .collect(),
            );
            assert_eq!(
                streamed,
                block_leaf_arena(&plan, &proof, k).expect("the arena"),
                "{name} leaf {k}: the streamed arena is the proof's"
            );
        }
        // A group handed over with another group's opening is refused or
        // differs: the arena is a function of each group's own share.
        if seen.len() >= 2 {
            let swapped = group_arena_words(
                &plan,
                0,
                &seen[0].tables,
                seen[0].argue.as_ref(),
                &seen[1].opening,
                seen[0].prepared.as_ref(),
            );
            assert!(
                swapped.as_ref().map_or(true, |w| *w != words[0]),
                "{name}: group 1's opening does not pass for group 0's"
            );
        }
    }
}

// ============================== the partition =============================

#[test]
fn the_group_partition_is_heaviest_first_onto_the_least_loaded_leaf() {
    // 9 → leaf 0, 7 → leaf 1, 5 → leaf 1 (7 < 9), 3 → leaf 0 (9 < 12),
    // 1 → leaf 0 (12 = 12, the lower leaf).
    let p = partition_groups(&[5, 9, 1, 7, 3], 2);
    assert_eq!(p.leaves(), &[vec![1, 2, 4], vec![0, 3]]);
    // Never more leaves than groups, and one leaf takes everything.
    assert_eq!(partition_groups(&[5, 9], 7).num_leaves(), 2);
    assert_eq!(partition_groups(&[5, 9, 1], 1).leaves(), &[vec![0, 1, 2]]);
}

/// ★ G3: no leaf over the cap. A group over it is refused (a group is atomic
/// for a leaf); without a fixed count, leaves are added from `⌈Σ/cap⌉` while
/// the heaviest is over it; and the block's own costs (FAST 421's W3 plan) keep
/// today's three leaves, so the default tree is unchanged.
#[test]
fn the_leaf_partition_keeps_every_leaf_under_the_cap() {
    let refused = leaf_partition(&[200, 10], None, 150);
    assert!(
        refused
            .as_ref()
            .is_err_and(|e| e.contains("group 0 costs 200")),
        "a group over the cap must be refused, got {refused:?}"
    );
    assert!(leaf_partition(&[200, 10], Some(2), 150).is_err());
    // ⌈300/150⌉ = 2 leaves would carry 200: one more leaf, 100 each.
    let p = leaf_partition(&[100, 100, 100], None, 150).expect("partitions");
    assert_eq!(p.num_leaves(), 3, "{:?}", p.leaves());
    // A fixed count (the tests', never a statement's) is the caller's.
    assert_eq!(
        leaf_partition(&[100, 100, 100], Some(1), 150)
            .expect("partitions")
            .num_leaves(),
        1
    );
    // A cost partition over one block's group costs, at the retired RPX leaf
    // cap of 279 k permutations (the function serves a fixed leaf count now).
    let block = [
        65825, 64356, 64405, 64474, 66559, 65146, 81574, 68482, 105991,
    ];
    let p = leaf_partition(&block, None, 279_000).expect("partitions");
    assert_eq!(p.leaves(), partition_groups(&block, 3).leaves());
    assert_eq!(p.leaves(), &[vec![1, 5, 8], vec![0, 2, 6], vec![3, 4, 7]]);
}

#[test]
fn the_group_partition_refuses_a_gap_an_overlap_and_an_empty_leaf() {
    assert!(BlockPartition::new(vec![vec![0], vec![2]], 3).is_err());
    assert!(BlockPartition::new(vec![vec![0, 1], vec![1, 2]], 3).is_err());
    assert!(BlockPartition::new(vec![vec![0, 1, 2], vec![]], 3).is_err());
    assert!(BlockPartition::new(vec![vec![3]], 3).is_err());
    assert!(BlockPartition::new(vec![vec![2, 0], vec![1]], 3).is_ok());
}

// ================================ the leaves ==============================

/// ★ Every leaf executes over an honest block proof — on the width-16 socket,
/// with no twelve-felt hash row — all of them publish the same id, state and
/// output — the plan's — and their shares cancel: the block's bus, checked
/// once over every table, closes across the leaves.
///
/// At blowup 4 ([`at_blowup_4`]): the test format's small stack makes a lone
/// wide table that production cannot form, and at blowup 2 its leaf rows
/// pass a leaf's room.
#[test]
#[ignore = "executes block leaves over small block proofs; box tier"]
fn whir_block_leaves_execute_and_close_the_bus() {
    let format = at_blowup_4(small_format());
    for (name, leaves) in [
        ("all_instructions_64", Some(3)),
        ("test_commit_4", Some(2)),
        ("test_keccak", None),
        ("test_commit_4", Some(1)),
    ] {
        let (elf, proof) = small_block(name, &format);
        let plan = plan_of(&elf, &proof, &format, leaves);
        let layout = plan.child_layout();
        println!(
            "WHIR BLOCK LEAVES {name}: {} groups, costs {:?}, partition {:?}, {} prepared",
            plan.num_groups(),
            plan.costs(),
            plan.partition().leaves(),
            plan.prepared().len()
        );
        for k in 0..plan.partition().num_leaves() {
            let program = plan.leaf_program(k).expect("the leaf emits");
            assert!(program.hash16(), "leaf {k} proves on the socket");
        }
        let words = leaves_words(&plan, &proof);
        let id = id_words(plan.id());
        let out: Vec<LfmWord> = out_halves(&proof.public_output)
            .into_iter()
            .map(base_word)
            .collect();
        let mut sum = FEE::zero();
        for w in &words {
            assert_eq!(w.len(), layout.total());
            assert_eq!(w[layout.id(0)], id[0]);
            assert_eq!(w[layout.id(1)], id[1]);
            assert_eq!(
                w[layout.state(0)],
                words[0][layout.state(0)],
                "one state across the leaves"
            );
            for (i, half) in out.iter().enumerate() {
                assert_eq!(w[layout.out_half(i)], *half);
            }
            sum += ext_of(&w[layout.sum()]);
        }
        assert_eq!(sum, FEE::zero(), "{name}: the leaves' shares cancel");
    }
}

/// ★ W1: the leaf's bus share ([`emit_share`]). Under the inverse form (the
/// default) a zero `q` has no satisfying assignment whatever `p` is —
/// `q · inv = 1` — which is the host's `contribution() = None` refusal. The
/// former `ediv(p, q)` (`LFM_WHIR_SHARE_INVERSE=0`) refuses `q = 0` only when
/// `p ≠ 0`: at `p = q = 0` its constraint `q · out = p` holds for every `out`
/// (the executor writes 1), so the share is free — the gap the inverse form
/// closes. The process default is checked to be the inverse form.
#[test]
fn a_zero_denominator_has_no_satisfying_assignment() {
    let run = |p: u64, q: u64, inverse: bool| {
        let mut b = LfmBuilder::new().with_wrap_hash(super::edsl::WrapHash::production());
        let a = b.declare_arena(2);
        let p_wire = b.hint_word(a, 0).as_ext();
        let q_wire = b.hint_word(a, 1).as_ext();
        let share = emit_share(&mut b, p_wire, q_wire, inverse);
        b.public(share.as_cell());
        let program = compile(b.finish());
        let words = vec![
            super::word::ext_word(&FEE::from(p)),
            super::word::ext_word(&FEE::from(q)),
        ];
        execute(&program, &[words], &crate::hash_pin::BLOCK_HASHER).map(|_| ())
    };
    for inverse in [false, true] {
        assert!(run(7, 3, inverse).is_ok(), "inverse {inverse}: 7/3");
        assert!(run(7, 0, inverse).is_err(), "inverse {inverse}: 7/0");
    }
    assert!(
        run(0, 0, true).is_err(),
        "the inverse form must refuse p = q = 0"
    );
    if std::env::var_os("LFM_WHIR_SHARE_INVERSE").is_none() {
        assert!(
            super::whir_block::share_inverse(),
            "the default share is the inverse form"
        );
    }
    assert!(
        run(0, 0, false).is_ok(),
        "the default ediv(p, q) executes p = q = 0 (its share is free): the gap"
    );
}

/// The production format with three polynomials a group: the dense fixture's
/// pages share groups with tables of their height.
fn dense_format() -> BlockFormat {
    BlockFormat {
        zf: ZfFormat::DEFAULT,
        group_polys: block_whir::BLOCK_GROUP_POLYS,
        max_groups: block_whir::BLOCK_MAX_GROUPS,
        prepared: true,
        argue: ArgueFormat::PerTable,
        group: multilinear::whir_chain::GroupFold::PerPoly,
        leaf_scale: crate::lfm::whir_block::LeafScale::Fixed,
        leaf_query: crate::lfm::whir_chain::LeafQuery::V1,
    }
}

fn dense_block_with(
    deviations: &block_whir::Deviations,
) -> (Vec<u8>, BlockWhirProof, Vec<block_whir::PreparedRoots>) {
    let elf = asm_elf_bytes("test_dense_pages");
    let roots = std::sync::Mutex::new(Vec::new());
    let proof = block_whir::prove_block_whir_observed_with(
        &elf,
        &[],
        &ProofOptions::default_test_options(),
        &dense_format(),
        &BlockOptions {
            max_rows: MaxRowsConfig::default(),
            keccak_rnd_rows_log2: 16,
            ecdas_rows_log2: block_whir::BLOCK_ECDAS_ROWS_LOG2,
            keccak_rows_log2: block_whir::BLOCK_KECCAK_ROWS_LOG2,
            ecsm_rows_log2: block_whir::BLOCK_ECSM_ROWS_LOG2,
            copy_rows_log2: block_whir::BLOCK_COPY_ROWS_LOG2,
            drop_levels: multilinear::whir_commit::TreeDrop::uniform(3),
            window_log2: None,
            stream_keccak_rnd: false,
            stream_memw_lt: false,
            drop_streamed_ops: false,
            layout_workers: 0,
            layout_ahead: Some(2),
            pack_rest_as_laid_out: false,
            // Every table packed on a card, so a box run walks the narrow path at a
            // test's size.
            narrow: stark::multilinear_block::Narrowing::Card { min_cells: 0 },
            upload_ahead: true,
            memlog: false,
            // The rest laid out in waves of 1 MiB: several at a test's size.
            finish_keccak_rnd_chunks: true,
            finish_cuts: true,
            finish_stream: true,
            rest_layout_bytes: Some(1 << 20),
            rest_build_bytes: None,
            pack_finished: true,
            gpack: true,
            spill: crate::block_whir::BlockSpillPolicy::Off,
            regen: None,
            phase_b_overlap: stark::multilinear_block::PhaseBOverlap::Off,
        },
        deviations,
        &|_, r| *roots.lock().expect("lock") = r.to_vec(),
    )
    .expect("the block proves")
    .0;
    let roots = roots.into_inner().expect("lock");
    (elf, proof, roots)
}

/// ★ A leaf refuses a prepared block opened at another table's point (a valid
/// opening of its stack, of the wrong claim), and with the prepared openings
/// left out (the mutation) it executes.
///
/// Then stacks committed wrongly, each opened consistently — two pages' blocks
/// swapped, a page's block holding the other page's columns, DECODE's block
/// another program's:
/// - refused by a leaf of the verifier's plan (it absorbs the stacks it
///   derives);
/// - refused by a leaf of a plan given the prover's own roots (the opening no
///   longer matches the tables' claims);
/// - executing only with those roots trusted AND the openings left out — the
///   two together bind the prepared columns to the program.
#[test]
#[ignore = "proves the dense fixture and executes its leaf; box tier"]
fn a_block_leaf_refuses_wrong_prepared_openings() {
    use block_whir::{Deviations, StackTamper};
    let format = dense_format();
    let opts = ProofOptions::default_test_options();
    let skip = LeafChecks {
        prepared: false,
        ..LeafChecks::ALL
    };
    let (elf, honest, _) = dense_block_with(&Deviations::default());
    let plan = plan_of(&elf, &honest, &format, Some(1));
    let tables: Vec<usize> = plan
        .prepared()
        .iter()
        .flat_map(|p| p.tables.iter().map(|&(_, air, _)| air))
        .collect();
    assert!(tables.len() >= 3, "DECODE and two pages: {tables:?}");
    run_leaf(&plan, &honest, 0, LeafChecks::ALL).expect("the honest leaf executes");

    // AIR indices: DECODE is the lowest prepared table, then the pages.
    let mut pages: Vec<usize> = tables.clone();
    pages.sort_unstable();
    let (a, b) = (pages[1], pages[2]);
    let group = |t: usize| {
        honest
            .groups
            .iter()
            .position(|g| g.contains(&(t as u32)))
            .expect("in a group")
    };
    let height = honest.table_num_vars[a];
    let other = honest.groups[group(a)]
        .iter()
        .map(|&t| t as usize)
        .find(|&t| t != a && honest.table_num_vars[t] == height)
        .expect("a table of the page's height in its group");
    let (_, wrong, _) = dense_block_with(&Deviations {
        prepared_points: vec![(a, other)],
        ..Default::default()
    });
    assert!(
        run_leaf(&plan, &wrong, 0, LeafChecks::ALL).is_err(),
        "wrong point"
    );
    assert!(
        run_leaf(&plan, &wrong, 0, skip).is_ok(),
        "wrong point, mutation"
    );

    let mut stack_tampers = vec![
        (
            "columns of another page",
            Deviations {
                prepared_stack: vec![StackTamper::ColumnsOf { table: a, from: b }],
                ..Default::default()
            },
        ),
        (
            "another program's DECODE",
            Deviations {
                other_prepared: true,
                ..Default::default()
            },
        ),
    ];
    if group(a) == group(b) {
        stack_tampers.push((
            "two pages swapped",
            Deviations {
                prepared_stack: vec![StackTamper::Swap(a, b)],
                ..Default::default()
            },
        ));
    }
    for (name, deviations) in &stack_tampers {
        let (_, wrong, prover_roots) = dense_block_with(deviations);
        assert!(
            run_leaf(&plan, &wrong, 0, LeafChecks::ALL).is_err(),
            "{name}: verifier's roots"
        );
        let trusting = WhirBlockPlan::derive_with(
            &elf,
            &opts,
            &format,
            wrong.statement(),
            Some(1),
            BLOCK_FAN_IN,
            Some(&prover_roots),
        )
        .expect("a plan over the prover's roots");
        assert!(
            run_leaf(&trusting, &wrong, 0, LeafChecks::ALL).is_err(),
            "{name}: prover's roots"
        );
        assert!(
            run_leaf(&trusting, &wrong, 0, skip).is_ok(),
            "{name}: the prover's roots trusted and the openings left out admit it"
        );
    }
}

// ================================= the nodes ==============================

/// A node's binding step over HINTED child words: the checks and publishes
/// `emit_block_node` runs after verifying its children, with the verification
/// left out so each check can be driven on its own.
fn bindings_program(
    layout: &BlockLayout,
    children: usize,
    top: bool,
    checks: BlockBindings,
) -> LfmProgram {
    let mut b = LfmBuilder::new().with_wrap_hash(super::edsl::WrapHash::production());
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

/// ★ Over REAL leaves' published words, every cross-child check refuses its
/// tamper and is the one refusing: with that check removed the same words bind.
/// Then the two block-level refusals: a leaf of another block's proof (another
/// statement, so another id and state), and a carrier that subtracted nothing
/// (the shares no longer cancel at the top).
#[test]
#[ignore = "executes block leaves over small block proofs; box tier"]
fn the_node_bindings_refuse_each_tamper_over_real_leaves() {
    let format = small_format();
    let (elf, proof) = small_block("test_commit_4", &format);
    let plan = plan_of(&elf, &proof, &format, Some(2));
    let layout = plan.child_layout();
    let honest = leaves_words(&plan, &proof);
    let top = run_bindings(&layout, &honest, true, BlockBindings::ALL).expect("honest leaves bind");
    assert_eq!(
        top[..],
        honest[0][..layout.total() - 1],
        "the top republishes the claim"
    );

    for (check, at) in [
        ("id", layout.id(1)),
        ("state", layout.state(0)),
        ("out", layout.out_half(0)),
        ("bus", layout.sum()),
    ] {
        let mut words = honest.clone();
        words[1][at][0] += FE::from(1u64);
        assert!(
            run_bindings(&layout, &words, true, BlockBindings::ALL).is_err(),
            "`{check}`: the tamper is refused"
        );
        assert!(
            run_bindings(&layout, &words, true, BlockBindings::without(check)).is_ok(),
            "`{check}`: with the check removed the tamper binds"
        );
    }

    // A leaf of another proof of the same program, packed into other groups:
    // another statement, so another id and another state. With its id words
    // made this block's, only the state tells it apart, and the state check is
    // what refuses it.
    let other_format = other_format();
    let (_, other) = small_block("test_commit_4", &other_format);
    let other_plan = plan_of(&elf, &other, &other_format, Some(2));
    assert_ne!(other_plan.id(), plan.id(), "the two statements differ");
    let foreign = leaves_words(&other_plan, &other);
    assert_ne!(foreign[1][layout.state(0)], honest[1][layout.state(0)]);
    let mut posing = foreign[1].clone();
    posing[layout.id(0)] = honest[0][layout.id(0)];
    posing[layout.id(1)] = honest[0][layout.id(1)];
    let mixed = vec![honest[0].clone(), posing];
    assert!(run_bindings(&layout, &mixed, false, BlockBindings::ALL).is_err());
    assert!(
        run_bindings(&layout, &mixed, false, BlockBindings::without("state")).is_ok(),
        "with the state check removed the foreign leaf binds"
    );

    // The carrier subtracting nothing: the shares sum to the target, not zero.
    let mut no_target = honest.clone();
    no_target[plan.carrier()] = run_leaf(
        &plan,
        &proof,
        plan.carrier(),
        LeafChecks {
            target: false,
            ..LeafChecks::ALL
        },
    )
    .expect("the leaf executes");
    assert!(run_bindings(&layout, &no_target, true, BlockBindings::ALL).is_err());
    assert!(run_bindings(&layout, &no_target, true, BlockBindings::without("bus")).is_ok());
}

// ================================= the tree ===============================

/// One tree program proved: the proof (what a parent's arena and the final
/// check read), the child a parent emits over, and its derived shape.
struct Proved {
    proof: LfmProof,
    child: RealChild,
    derived: DerivedChild,
}

fn prove_tree_program(
    label: &str,
    program: &LfmProgram,
    arenas: &[Vec<LfmWord>],
    words: usize,
) -> Result<(Proved, f64), String> {
    let opts = aggregation_wrap_options();
    let t = std::time::Instant::now();
    let artifacts = artifacts_of(program, &opts);
    let t_artifacts = t.elapsed().as_secs_f64();
    let derived = super::whir_block::derived_of(&artifacts, &opts, words)?;
    let t = std::time::Instant::now();
    let proof = lfm_prove_under::<BlockFamily>(program, &artifacts, arenas, &opts)
        .map_err(|e| format!("{label}: {e:?}"))?;
    let t_prove = t.elapsed().as_secs_f64();
    let (child, verify) =
        super::harvest::harvest_child_verified_under::<BlockFamily>(artifacts, opts, &proof)?;
    println!(
        "   {label} TIMING: {} instructions · build_artifacts {t_artifacts:.2}s · prove {t_prove:.2}s · harvest verify {verify:.2}s",
        program.instrs.len()
    );
    Ok((
        Proved {
            proof,
            child,
            derived,
        },
        t_artifacts + t_prove,
    ))
}

/// The tree over `plan`'s leaves to its top. Returns the top's proof and, per
/// level (the leaves first), its wall and the sum of its programs' artifact and
/// prove seconds.
fn compose(
    plan: &WhirBlockPlan,
    proof: &BlockWhirProof,
    tag: &str,
) -> Result<(LfmProof, Vec<(f64, f64)>), String> {
    let words = plan.child_layout().total();
    let mut walls = Vec::new();
    let t = std::time::Instant::now();
    let mut busy = 0.0;
    let mut level: Vec<Proved> = Vec::with_capacity(plan.partition().num_leaves());
    for k in 0..plan.partition().num_leaves() {
        let te = std::time::Instant::now();
        let program = plan.leaf_program(k)?;
        let arenas = block_leaf_arena(plan, proof, k)?;
        println!(
            "   {tag} LEAF {k}: groups {:?} · emitted in {:.2}s · arena {} words",
            plan.partition().leaf(k),
            te.elapsed().as_secs_f64(),
            arenas[0].len()
        );
        let (proved, secs) =
            prove_tree_program(&format!("{tag} LEAF {k}"), &program, &arenas, words)?;
        busy += secs;
        level.push(proved);
    }
    let wall = t.elapsed().as_secs_f64();
    println!(
        "   {tag} LEVEL 0: {} leaves in {wall:.2}s (artifacts + prove {busy:.2}s)",
        level.len()
    );
    walls.push((wall, busy));
    let levels = plan.levels();
    for (lv, arities) in levels.iter().enumerate() {
        let top = lv + 1 == levels.len();
        let t = std::time::Instant::now();
        let mut busy = 0.0;
        let mut next = Vec::with_capacity(arities.arities.len());
        let mut rest = level.into_iter();
        for (j, &a) in arities.arities.iter().enumerate() {
            let kids: Vec<Proved> = rest.by_ref().take(a).collect();
            let arenas: Vec<Vec<LfmWord>> = kids
                .iter()
                .flat_map(|k| child_arena_words(&k.child))
                .collect();
            let derived: Vec<&DerivedChild> = kids.iter().map(|k| &k.derived).collect();
            let program = plan.node_program(&derived, top)?;
            let (proved, secs) = prove_tree_program(
                &format!("{tag} L{}N{j}{}", lv + 1, if top { " TOP" } else { "" }),
                &program,
                &arenas,
                words,
            )?;
            busy += secs;
            next.push(proved);
        }
        let wall = t.elapsed().as_secs_f64();
        println!(
            "   {tag} LEVEL {}: {} node(s) in {wall:.2}s (artifacts + prove {busy:.2}s)",
            lv + 1,
            next.len(),
        );
        walls.push((wall, busy));
        level = next;
    }
    let [top] = <[Proved; 1]>::try_from(level).map_err(|l| format!("{} tops", l.len()))?;
    Ok((top.proof, walls))
}

/// ★ ECDAS cut through the block's recursion: test_ecsm_multi with ECDAS in
/// 16-row tables (three at the default, the fourth of padding rows alone not
/// built; the 0xABCDEF call through the three) proves as a block
/// whose tree the verifier accepts. The same statement with one ECDAS table
/// stated over [`block_whir::BLOCK_ECDAS_MAX_VARS`] is refused before any
/// program is derived.
///
/// At blowup 4 ([`at_blowup_4`]): the test format's small stack makes a lone
/// wide table that production cannot form, and at blowup 2 its leaf rows
/// pass a leaf's room.
#[test]
#[ignore = "proves block leaves and nodes over a small block; box tier"]
fn the_whir_block_tree_verifies_a_split_ecdas() {
    super::device_permit::arm(1);
    let format = at_blowup_4(small_format());
    let (elf, proof) = small_block_at("test_ecsm_multi", &format, 4);
    assert_eq!(proof.table_counts.ecdas, 3, "three ECDAS tables");
    let opts = ProofOptions::default_test_options();
    let wrap = aggregation_wrap_options();
    let plan = plan_of(&elf, &proof, &format, None);
    let (top, _) = compose(&plan, &proof, "ECDAS TREE").expect("the honest tree proves");
    let verify = |statement: block_whir::BlockStatement<'_>| {
        verify_block_tree_under(
            &elf,
            &opts,
            &format,
            statement,
            None,
            BLOCK_FAN_IN,
            &wrap,
            &top,
        )
    };
    verify(proof.statement()).expect("the block's verifier accepts the split ECDAS");

    let mut tall = proof.statement().to_owned();
    let (_, range, cap) = block_whir::chunked_table_ranges(&tall.table_counts)
        .into_iter()
        .find(|(name, ..)| *name == "ECDAS")
        .expect("the ECDAS range");
    tall.table_num_vars[range.start] = (cap + 1) as u8;
    let refused = verify(tall.view());
    assert!(
        refused.is_err(),
        "an ECDAS table stated over its cap must be refused"
    );
    println!(
        "WHIR BLOCK TREE ECDAS SPLIT: {} groups, 3 ECDAS tables, accepted; an ECDAS over its cap \
         refused ({})",
        plan.num_groups(),
        refused.err().unwrap_or_default()
    );
}

/// ★ KECCAK, ECSM and COPY cut through the block's recursion:
/// test_keccak_multi's three permutations and test_ecsm_multi's three scalar
/// multiplications, one row a table (three tables each at the default: the
/// three calls, the padding row's table not built), and test_copy_ecalls'
/// 17 COPY rows, one row a table (17 tables, calls cut inside), prove as
/// blocks whose trees the verifier derives and accepts. Each statement with
/// its first table stated over its cap ([`block_whir::BLOCK_KECCAK_MAX_VARS`],
/// [`block_whir::BLOCK_ECSM_MAX_VARS`], [`block_whir::BLOCK_COPY_MAX_VARS`])
/// is refused before any program is derived.
///
/// At blowup 4 ([`at_blowup_4`]): the test format's small stack makes a lone
/// wide table that production cannot form, and at blowup 2 its leaf rows
/// pass a leaf's room.
#[test]
#[ignore = "proves three small blocks and their trees; box tier"]
fn the_whir_block_tree_verifies_split_keccak_and_ecsm() {
    super::device_permit::arm(1);
    let format = at_blowup_4(small_format());
    let opts = ProofOptions::default_test_options();
    let wrap = aggregation_wrap_options();
    for (guest, name, tables) in [
        ("test_keccak_multi", "KECCAK", 3),
        ("test_ecsm_multi", "ECSM", 3),
        ("test_copy_ecalls", "COPY", 17),
    ] {
        let (elf, proof) = small_block_cut(guest, &format, |o| match name {
            "KECCAK" => o.keccak_rows_log2 = 0,
            "COPY" => o.copy_rows_log2 = 0,
            _ => o.ecsm_rows_log2 = 0,
        });
        let (_, range, cap) = block_whir::chunked_table_ranges(&proof.table_counts)
            .into_iter()
            .find(|(n, ..)| *n == name)
            .expect("the table's range");
        assert_eq!(range.len(), tables, "{name}: one row a table");
        let plan = plan_of(&elf, &proof, &format, None);
        let (top, _) =
            compose(&plan, &proof, &format!("{name} TREE")).expect("the honest tree proves");
        let verify = |statement: block_whir::BlockStatement<'_>| {
            verify_block_tree_under(
                &elf,
                &opts,
                &format,
                statement,
                None,
                BLOCK_FAN_IN,
                &wrap,
                &top,
            )
        };
        verify(proof.statement())
            .unwrap_or_else(|e| panic!("the block's verifier accepts the split {name}: {e}"));

        let mut tall = proof.statement().to_owned();
        tall.table_num_vars[range.start] = (cap + 1) as u8;
        let refused = verify(tall.view());
        assert!(
            refused.is_err(),
            "a {name} table stated over its cap must be refused"
        );
        println!(
            "WHIR BLOCK TREE {name} SPLIT: {} groups, {tables} {name} tables, accepted; a {name} \
             over its cap refused ({})",
            plan.num_groups(),
            refused.err().unwrap_or_default()
        );
    }
}

/// The verifier's derive windows: at most `n` items at once, every item once,
/// in order, and the first error stops the level.
#[test]
fn a_derive_window_holds_its_bound_and_its_order() {
    use std::sync::atomic::{AtomicUsize, Ordering};
    let items: Vec<usize> = (0..23).collect();
    for window in [None, Some(1), Some(2), Some(5), Some(64)] {
        let inside = AtomicUsize::new(0);
        let most = AtomicUsize::new(0);
        let done = super::whir_block::map_in_windows(&items, window, |&i| {
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
    let failed = super::whir_block::map_in_windows(&items, Some(4), |&i| {
        if i == 9 {
            Err(format!("item {i}"))
        } else {
            Ok(i)
        }
    });
    assert_eq!(failed, Err("item 9".to_string()));
}

/// The derive's knobs read a positive window and `level`, and refuse nonsense
/// with an error, not a panic.
#[test]
fn the_derive_knobs_read_their_values_and_refuse_nonsense() {
    use super::whir_block::{parse_derive_builds, parse_derive_hold};
    assert_eq!(parse_derive_builds(None), Ok(None));
    assert_eq!(parse_derive_builds(Some(" ")), Ok(None));
    assert_eq!(parse_derive_builds(Some("8")), Ok(Some(8)));
    for bad in ["0", "-1", "eight"] {
        assert!(parse_derive_builds(Some(bad)).is_err(), "{bad}");
    }
    assert_eq!(parse_derive_hold(None), Ok(false));
    assert_eq!(parse_derive_hold(Some("level")), Ok(true));
    assert!(parse_derive_hold(Some("all")).is_err());
}

/// The leaf scale's knob reads nothing as the default (doubled from 10
/// leaves), `off` as the single heights and a positive count as the
/// threshold, refuses anything else with an error, and the production format
/// holds what the environment says.
#[test]
fn the_leaf_scale_knob_reads_off_or_a_count() {
    use super::whir_block::{LeafScale, leaf_scale_from_env, leaf_scale_setting};
    assert_eq!(LeafScale::DEFAULT, LeafScale::DoubleFrom(10));
    for unset in [None, Some(""), Some(" ")] {
        assert_eq!(
            leaf_scale_setting(unset),
            Ok(LeafScale::DEFAULT),
            "{unset:?}"
        );
    }
    for off in [Some("off"), Some(" off ")] {
        assert_eq!(leaf_scale_setting(off), Ok(LeafScale::Fixed), "{off:?}");
    }
    assert_eq!(
        leaf_scale_setting(Some("12")),
        Ok(LeafScale::DoubleFrom(12))
    );
    assert_eq!(
        leaf_scale_setting(Some(" 10 ")),
        Ok(LeafScale::DoubleFrom(10))
    );
    assert_eq!(leaf_scale_setting(Some("1")), Ok(LeafScale::DoubleFrom(1)));
    for bad in ["0", "-1", "twelve", "on", "12 leaves", "OFF"] {
        let err = leaf_scale_setting(Some(bad)).expect_err(bad);
        assert!(
            err.contains(super::whir_block::LEAF_DOUBLE_FROM_ENV),
            "{err}"
        );
    }
    assert_eq!(
        Ok(BlockFormat::production().leaf_scale),
        leaf_scale_from_env(),
        "the production format holds the environment's scale"
    );
}

/// ★ The verifier's lean derive ([`WhirBlockPlan::derive_top_in`]: each
/// program and its artifacts dropped once its child's shape is derived, the
/// leaves in windows) derives the held derive's top ([`WhirBlockPlan::
/// programs`]) byte for byte, at a window of 1, of 8 and the whole level, over
/// a tree with a node level below the top.
#[test]
#[ignore = "proves a small block's base; box tier"]
fn the_lean_derive_is_the_held_derives_top_at_every_window() {
    let format = small_format();
    let (elf, proof) = small_block("test_commit_4", &format);
    let opts = ProofOptions::default_test_options();
    let wrap = aggregation_wrap_options();
    // The group count is the statement's, whatever the leaf count. Under the
    // chip bound (`None`) the plan refuses this block at blowup 2: its group 0
    // spends more lanes rows than a single leaf holds.
    let groups = plan_of(&elf, &proof, &format, Some(1)).num_groups();
    let leaves = groups.min(5);
    assert!(
        leaves >= 3,
        "{groups} groups: too few for a node level below the top"
    );
    let plan = WhirBlockPlan::derive_with(
        &elf,
        &opts,
        &format,
        proof.statement(),
        Some(leaves),
        2,
        None,
    )
    .expect("the plan derives");
    assert!(plan.levels().len() >= 2, "a node level below the top");
    let mut held = plan.programs(&wrap).expect("the held derive");
    let held_top = held
        .levels
        .pop()
        .and_then(|mut level| level.pop())
        .expect("a top")
        .artifacts;
    for window in [Some(1), Some(8), None] {
        let lean = plan.derive_top_in(&wrap, window).expect("the lean derive");
        assert_eq!(
            lean, held_top,
            "window {window:?}: the lean top is the held top"
        );
    }
    println!(
        "LEAN DERIVE IDENTITY: {leaves} leaves, {} node levels, the top equal at windows 1, 8 and the whole level",
        plan.levels().len()
    );
}

/// ★ Under the lean derive the verifier still refuses: a plan with one leaf
/// over a tampered prepared root derives another top, against which the
/// honest top proof does not verify (the leaf case), and a top proof with one
/// published word changed is refused by the block's verifier (the node case),
/// at every window; the honest top verifies at every window.
#[test]
#[ignore = "proves block leaves and nodes over a small block; box tier"]
fn the_lean_derive_still_refuses_a_tampered_leaf_and_node() {
    super::device_permit::arm(1);
    let format = small_format();
    let (elf, proof) = small_block("test_commit_4", &format);
    let opts = ProofOptions::default_test_options();
    let wrap = aggregation_wrap_options();
    let plan = plan_of(&elf, &proof, &format, Some(2));
    let (top, _) = compose(&plan, &proof, "LEAN").expect("the honest tree proves");
    let mut roots: Vec<block_whir::PreparedRoots> = plan
        .prepared()
        .iter()
        .map(|p| (p.group, p.roots.clone()))
        .collect();
    roots[0].1[0][0] ^= 1;
    let bad = WhirBlockPlan::derive_with(
        &elf,
        &opts,
        &format,
        proof.statement(),
        Some(2),
        BLOCK_FAN_IN,
        Some(&roots),
    )
    .expect("a plan over a tampered root");
    for window in [Some(1), Some(8), None] {
        let honest = plan.derive_top_in(&wrap, window).expect("the honest top");
        assert!(
            super::proof::verify_against_artifacts_under::<BlockFamily>(
                &honest,
                &top.proof,
                &top.public_words,
                &wrap
            ),
            "window {window:?}: the honest top verifies"
        );
        let other = bad
            .derive_top_in(&wrap, window)
            .expect("the tampered plan's top");
        assert_ne!(other.program_id, honest.program_id, "window {window:?}");
        assert!(
            !super::proof::verify_against_artifacts_under::<BlockFamily>(
                &other,
                &top.proof,
                &top.public_words,
                &wrap
            ),
            "window {window:?}: the honest top does not verify against a tampered leaf's tree"
        );
    }
    let mut words = top.public_words.clone();
    words[0].1[0] += FE::from(1u64);
    let node = LfmProof {
        proof: top.proof.clone(),
        public_words: words,
    };
    assert!(
        verify_block_tree_under(
            &elf,
            &opts,
            &format,
            proof.statement(),
            Some(2),
            BLOCK_FAN_IN,
            &wrap,
            &node,
        )
        .is_err(),
        "a top proof with one published word changed is refused"
    );
    assert!(
        verify_block_tree_under(
            &elf,
            &opts,
            &format,
            proof.statement(),
            Some(2),
            BLOCK_FAN_IN,
            &wrap,
            &top,
        )
        .is_ok(),
        "the honest top is accepted by the block's verifier"
    );
}

/// ★ The tree over a small block proves to a top, and the block's verifier —
/// which derives the top program from the ELF and the statement, never from
/// the proof — accepts it.
///
/// Two trees the plan would not derive:
/// - the same groups with the leaves' lists swapped (the carrier now holds the
///   other half): every group once and the bus closes, so it proves, and the
///   verifier refuses it; checked against its OWN top program it is accepted,
///   so the refusal is the derived program's identity;
/// - a group verified twice and another never: refused, at the latest by the
///   verifier.
#[test]
#[ignore = "proves block leaves and nodes over a small block; box tier"]
fn the_whir_block_tree_proves_to_the_derived_top() {
    super::device_permit::arm(1);
    let format = small_format();
    let (elf, proof) = small_block("test_commit_4", &format);
    let opts = ProofOptions::default_test_options();
    let wrap = aggregation_wrap_options();
    let plan = plan_of(&elf, &proof, &format, Some(2));
    let (top, _) = compose(&plan, &proof, "TREE").expect("the honest tree proves");
    verify_block_tree_under(
        &elf,
        &opts,
        &format,
        proof.statement(),
        Some(2),
        BLOCK_FAN_IN,
        &wrap,
        &top,
    )
    .expect("the block's verifier accepts the honest top");
    // The block's base is its proof family's: under RPX's the verifier refuses
    // the honest top, and no RPX plan derives at all.
    let rpx = BlockFormat {
        zf: format.zf.with_base(stark::proof::options::BaseFormat::RPX),
        ..format
    };
    assert!(
        verify_block_tree_under(
            &elf,
            &opts,
            &rpx,
            proof.statement(),
            Some(2),
            BLOCK_FAN_IN,
            &wrap,
            &top,
        )
        .is_err(),
        "the top under an RPX base"
    );
    assert!(
        WhirBlockPlan::derive(&elf, &opts, &rpx, proof.statement(), Some(2)).is_err(),
        "no plan under an RPX base"
    );

    // The prover's roots shortcut only the prover's side. A plan built from
    // tampered ones holds other leaf programs, so another top; its leaves
    // refuse the honest proof (every leaf absorbs the roots), so no tree can be
    // built over them; and the plan built from the true roots is the
    // verifier's own.
    let roots: Vec<block_whir::PreparedRoots> = plan
        .prepared()
        .iter()
        .map(|p| (p.group, p.roots.clone()))
        .collect();
    let mut tampered = roots.clone();
    tampered[0].1[0][0] ^= 1;
    let from = |roots: &[block_whir::PreparedRoots]| {
        WhirBlockPlan::derive_with(
            &elf,
            &opts,
            &format,
            proof.statement(),
            Some(2),
            BLOCK_FAN_IN,
            Some(roots),
        )
        .expect("a plan over supplied roots")
    };
    let honest_top = plan.derive_top(&wrap).expect("the top derives").program_id;
    assert_eq!(
        from(&roots).derive_top(&wrap).expect("top").program_id,
        honest_top
    );
    let bad = from(&tampered);
    assert_ne!(bad.derive_top(&wrap).expect("top").program_id, honest_top);
    for k in 0..bad.partition().num_leaves() {
        assert!(
            run_leaf(&bad, &proof, k, LeafChecks::ALL).is_err(),
            "leaf {k} over tampered roots"
        );
    }

    let groups = plan.num_groups();
    assert!(groups >= 2, "the block spans at least two groups");
    let lists = plan.partition().leaves().to_vec();
    let swapped = plan_of(&elf, &proof, &format, Some(2)).with_partition(
        BlockPartition::new(vec![lists[1].clone(), lists[0].clone()], groups)
            .expect("still a partition"),
    );
    let (other, _) =
        compose(&swapped, &proof, "SWAPPED").expect("a tree over another partition proves");
    assert!(
        verify_block_tree_under(
            &elf,
            &opts,
            &format,
            proof.statement(),
            Some(2),
            BLOCK_FAN_IN,
            &wrap,
            &other
        )
        .is_err(),
        "a tree over another partition is refused"
    );
    let own = swapped.derive_top(&wrap).expect("its own top derives");
    assert!(
        super::proof::verify_against_artifacts_under::<BlockFamily>(
            &own,
            &other.proof,
            &other.public_words,
            &wrap
        ),
        "against its own top it verifies: the derived program's identity is what refuses it"
    );

    // A carrier that subtracts nothing: both leaves prove, and the top node
    // cannot — unless its bus check is removed.
    let words = plan.child_layout().total();
    let leaves: Vec<Proved> = (0..2)
        .map(|k| {
            let checks = if k == plan.carrier() {
                LeafChecks {
                    target: false,
                    ..LeafChecks::ALL
                }
            } else {
                LeafChecks::ALL
            };
            let program = leaf_program_with(&plan, k, checks).expect("the leaf emits");
            let arenas = block_leaf_arena(&plan, &proof, k).expect("the arena");
            prove_tree_program(&format!("NO-TARGET LEAF {k}"), &program, &arenas, words)
                .expect("each leaf proves on its own")
                .0
        })
        .collect();
    let derived: Vec<_> = leaves.iter().map(|l| l.derived.shape()).collect();
    let arenas: Vec<Vec<LfmWord>> = leaves
        .iter()
        .flat_map(|l| child_arena_words(&l.child))
        .collect();
    let node = |checks: BlockBindings| -> LfmProgram {
        let mut b = LfmBuilder::new().with_wrap_hash(<BlockFamily as ProofFamily>::WRAP);
        emit_block_node_with(
            &mut b,
            &BlockNodeInputs {
                children: &derived,
                layout: plan.child_layout(),
                top: true,
            },
            checks,
        );
        compile(b.finish())
    };
    assert!(
        prove_tree_program("NO-TARGET TOP", &node(BlockBindings::ALL), &arenas, words).is_err(),
        "the top refuses shares that do not cancel"
    );
    assert!(
        prove_tree_program(
            "NO-TARGET TOP (no bus check)",
            &node(BlockBindings::without("bus")),
            &arenas,
            words
        )
        .is_ok(),
        "with the bus check removed the same leaves close a top"
    );

    let skewed = plan_of(&elf, &proof, &format, Some(2)).with_partition(
        BlockPartition::unvalidated(vec![vec![0], (0..groups - 1).collect()], groups),
    );
    match compose(&skewed, &proof, "SKEWED") {
        Err(e) => println!("   SKEWED: refused while proving: {e}"),
        Ok((bad, _)) => assert!(
            verify_block_tree_under(
                &elf,
                &opts,
                &format,
                proof.statement(),
                Some(2),
                BLOCK_FAN_IN,
                &wrap,
                &bad
            )
            .is_err(),
            "a tree that verifies a group twice and another never is refused"
        ),
    }
}

/// The leaf query knob reads `lean` (or nothing) and `v1`, and refuses
/// anything else with an error; the production format holds what the
/// environment says.
#[test]
fn the_leaf_query_knob_reads_v1_or_lean() {
    use super::whir_block::{LEAF_QUERY_ENV, leaf_query_from_env, leaf_query_setting};
    use super::whir_chain::LeafQuery;
    for lean in [None, Some(""), Some("lean"), Some(" lean ")] {
        assert_eq!(leaf_query_setting(lean), Ok(LeafQuery::Lean), "{lean:?}");
    }
    assert_eq!(leaf_query_setting(Some(" v1 ")), Ok(LeafQuery::V1));
    for bad in ["Lean", "on", "1", "v2"] {
        let err = leaf_query_setting(Some(bad)).expect_err(bad);
        assert!(err.contains(LEAF_QUERY_ENV), "{err}");
    }
    assert_eq!(
        Ok(BlockFormat::production().leaf_query),
        leaf_query_from_env(),
        "the production format holds the environment's check"
    );
}

/// The leaves check lean by default: the type's default, the knob unset, the
/// production format with the knob unset, and both block commands' flag
/// (`bin/cli`). `v1` stays a named choice, and every chain outside a block
/// leaf keeps the per-query check (`ChainShape::new`).
#[test]
fn the_leaves_check_lean_by_default() {
    use super::whir_block::{LEAF_QUERY_ENV, leaf_query_setting};
    use super::whir_chain::{ChainShape, LeafQuery};
    assert_eq!(LeafQuery::default(), LeafQuery::Lean);
    assert_eq!(leaf_query_setting(None), Ok(LeafQuery::Lean));
    assert_eq!(leaf_query_setting(Some("v1")), Ok(LeafQuery::V1));
    if std::env::var_os(LEAF_QUERY_ENV).is_none() {
        assert_eq!(BlockFormat::production().leaf_query, LeafQuery::Lean);
    }
    let config = BlockFormat::production().chain_config(&[(1, 12)]);
    assert_eq!(ChainShape::new(&config, 12).query, LeafQuery::V1);
}

/// ★ The leaf scale is the verifier's, both ways, on a small block whose
/// groups each need a leaf of their own under a leaf cap of the heaviest
/// group's socket rows ([`WhirBlockPlan::derive_capped`]): `n` leaves at the
/// single heights. At [`LeafScale::DoubleFrom`]`(n)` the plan doubles them,
/// to fewer leaves; at `DoubleFrom(n + 1)` it keeps the single partition.
/// Both trees prove, and the block's verifier, which derives the partition
/// from the statement under ITS scale:
/// - refuses the doubled tree under the single heights, and the single tree
///   under `DoubleFrom(n)`: the cross-mode refusals;
/// - refuses the doubled tree under `DoubleFrom(n + 1)`: a prover cannot
///   double below the threshold;
/// - accepts each tree under its own scale, and the single tree under
///   `DoubleFrom(n + 1)`: below the threshold the tree is the single one.
///
/// The refusals come first, each the one a verifier-side mutant turns red
/// (one that always doubles, one that never does, one that doubles from one
/// leaf fewer). Against its own derived top the doubled tree verifies: the
/// refusal is the derived program's identity.
///
/// At blowup 4, the rate this block's shape is sized at: at blowup 2 the
/// small format's group 0 spends more lanes rows than a single leaf holds
/// (1,051,383 against 1,048,482 at Q 229), so every scale's partition refuses
/// the block.
#[test]
#[ignore = "proves two trees over a small block; box tier"]
fn the_leaf_scale_is_the_verifiers_both_ways() {
    use super::whir_block::{LeafScale, verify_block_tree_capped};
    super::device_permit::arm(1);
    let format = BlockFormat {
        zf: ZfFormat {
            whir_blowup: 4,
            ..small_format().zf
        },
        ..small_format()
    };
    let (elf, proof) = small_block("test_commit_4", &format);
    let opts = ProofOptions::default_test_options();
    let wrap = aggregation_wrap_options();
    let probe = plan_of(&elf, &proof, &format, Some(1));
    let (loads, front) = probe.chip_loads();
    let cap = front.hash + loads.iter().map(|l| l.hash).max().expect("groups");
    let at = |leaf_scale: LeafScale| BlockFormat {
        leaf_scale,
        ..format
    };
    let plan_at = |scale: LeafScale| {
        WhirBlockPlan::derive_capped(
            &elf,
            &opts,
            &at(scale),
            proof.statement(),
            None,
            BLOCK_FAN_IN,
            None,
            cap,
        )
        .expect("the plan derives")
    };
    let single = plan_at(LeafScale::Fixed);
    let n = single.partition().num_leaves();
    assert!(n >= 2, "{n} leaves at a cap of {cap} socket rows");
    let doubled = plan_at(LeafScale::DoubleFrom(n));
    let m = doubled.partition().num_leaves();
    let bound = doubled.leaf_bound().expect("the rule's partition");
    assert!(
        bound.doubled && bound.single_leaves == n && m < n,
        "doubled from {n}: {m} leaves, {bound:?}"
    );
    let below = plan_at(LeafScale::DoubleFrom(n + 1));
    assert_eq!(below.partition(), single.partition(), "below the threshold");
    println!(
        "   LEAF SCALE: {} groups · cap {cap} · {n} single leaves {:?} · {m} doubled {:?}",
        single.num_groups(),
        single.partition().leaves(),
        doubled.partition().leaves()
    );
    let (top_single, _) = compose(&single, &proof, "SINGLE").expect("the single tree proves");
    let (top_doubled, _) = compose(&doubled, &proof, "DOUBLED").expect("the doubled tree proves");
    let verify = |scale: LeafScale, top: &LfmProof| {
        verify_block_tree_capped(
            &elf,
            &opts,
            &at(scale),
            proof.statement(),
            BLOCK_FAN_IN,
            &wrap,
            top,
            cap,
        )
    };
    assert!(
        verify(LeafScale::Fixed, &top_doubled).is_err(),
        "the doubled tree under the single heights"
    );
    assert!(
        verify(LeafScale::DoubleFrom(n), &top_single).is_err(),
        "the single tree under doubled leaves from {n}"
    );
    assert!(
        verify(LeafScale::DoubleFrom(n + 1), &top_doubled).is_err(),
        "the doubled tree below the threshold {}",
        n + 1
    );
    verify(LeafScale::Fixed, &top_single).expect("the single tree under its own scale");
    verify(LeafScale::DoubleFrom(n), &top_doubled).expect("the doubled tree under its own scale");
    verify(LeafScale::DoubleFrom(n + 1), &top_single)
        .expect("below the threshold the single tree is the verifier's");
    let own = doubled.derive_top(&wrap).expect("its own top derives");
    assert!(
        super::proof::verify_against_artifacts_under::<BlockFamily>(
            &own,
            &top_doubled.proof,
            &top_doubled.public_words,
            &wrap
        ),
        "against its own top the doubled tree verifies"
    );
}

// ===================== the lean query check over a block =====================

/// `format` with its leaves checking each chain's queries the lean way
/// ([`LeafQuery::Lean`](super::whir_chain::LeafQuery::Lean)).
fn lean_format(format: BlockFormat) -> BlockFormat {
    BlockFormat {
        leaf_query: super::whir_chain::LeafQuery::Lean,
        ..format
    }
}

/// ★ Under the lean query check every leaf of a small block executes over the
/// honest proof, per polynomial and with grouped round-0 trees, and the
/// leaves' shares cancel; its leaf programs are not V1's (fewer `Select`s);
/// and a round-0 Merkle sibling or value tampered at query 1 is refused by the
/// leaf holding the group.
#[test]
#[ignore = "executes block leaves over small block proofs; box tier"]
fn lean_block_leaves_execute_and_refuse_a_tampered_witness() {
    use super::instr::Instr;
    use multilinear::whir_chain::RoundOpenings;
    let indexed = indexed_format(crate::hash_pin::BLOCK_BASE, ArgueFormat::PerTable);
    for (format, name) in [
        (small_format(), "test_commit_4"),
        (indexed, "all_instructions_64"),
    ] {
        let lean = lean_format(format);
        let (elf, proof) = small_block(name, &lean);
        let plan = plan_of(&elf, &proof, &lean, Some(2));
        let words = leaves_words(&plan, &proof);
        let layout = plan.child_layout();
        let sum = words
            .iter()
            .fold(FEE::zero(), |acc, w| acc + ext_of(&w[layout.sum()]));
        assert_eq!(sum, FEE::zero(), "{name}: the leaves' shares cancel");
        let v1 = plan_of(&elf, &proof, &format, Some(2));
        let selects = |p: &WhirBlockPlan| {
            let program = p.leaf_program(0).expect("the leaf emits");
            program
                .instrs
                .iter()
                .filter(|i| matches!(i, Instr::Select { .. }))
                .count()
        };
        assert!(
            selects(&plan) < selects(&v1),
            "{name}: lean {} Selects against V1 {}",
            selects(&plan),
            selects(&v1)
        );

        let g = 0;
        let k = (0..plan.partition().num_leaves())
            .find(|&k| plan.partition().leaf(k).contains(&g))
            .expect("a leaf holds the group");
        run_leaf(&plan, &proof, k, LeafChecks::ALL).expect("the honest leaf executes");
        for value in [false, true] {
            let mut forged = proof.clone();
            match &mut forged.proof.columns[g].polys[0].rounds[0].openings {
                RoundOpenings::Base(p) if value => p.current[1].values[0] += FE::one(),
                RoundOpenings::Base(p) => p.current[1].proof.merkle_path[0][0] ^= 1,
                RoundOpenings::BaseGrouped(p) if value => p.current[1][0].values[0] += FE::one(),
                RoundOpenings::BaseGrouped(p) => p.current[1][0].proof.merkle_path[0][0] ^= 1,
                RoundOpenings::Extension(_) => unreachable!("round 0 is base-valued"),
            }
            assert!(
                run_leaf(&plan, &forged, k, LeafChecks::ALL).is_err(),
                "{name}: a round-0 {} at query 1",
                if value { "value" } else { "sibling" }
            );
        }
    }
}

/// ★ The leaves' query check is the verifier's, both ways, on proved trees: a
/// tree proved under `Lean` is refused by a `V1` verifier and a `V1` tree by a
/// `Lean` one, and each verifies under its own check. Each refusal is the one
/// a verifier-side mutant turns red (one that always takes the lean leaf, one
/// that never does).
#[test]
#[ignore = "proves two trees over a small block; box tier"]
fn the_leaf_query_is_the_verifiers_both_ways() {
    use super::whir_block::{LEAF_P1_CAP, verify_block_tree_capped};
    use super::whir_chain::LeafQuery;
    super::device_permit::arm(1);
    // At blowup 4 ([`at_blowup_4`]): at blowup 2 this small format's V1
    // groups pass a single leaf's room, so the rule's partition refuses.
    let format = at_blowup_4(small_format());
    let (elf, proof) = small_block("test_commit_4", &format);
    let opts = ProofOptions::default_test_options();
    let wrap = aggregation_wrap_options();
    let at = |leaf_query: LeafQuery| BlockFormat {
        leaf_query,
        ..format
    };
    let plan_at = |query: LeafQuery| plan_of(&elf, &proof, &at(query), None);
    let (v1, lean) = (plan_at(LeafQuery::V1), plan_at(LeafQuery::Lean));
    println!(
        "   LEAF QUERY: {} groups · V1 {:?} · lean {:?}",
        v1.num_groups(),
        v1.partition().leaves(),
        lean.partition().leaves()
    );
    let (top_v1, _) = compose(&v1, &proof, "V1").expect("the V1 tree proves");
    let (top_lean, _) = compose(&lean, &proof, "LEAN").expect("the lean tree proves");
    let verify = |query: LeafQuery, top: &LfmProof| {
        verify_block_tree_capped(
            &elf,
            &opts,
            &at(query),
            proof.statement(),
            BLOCK_FAN_IN,
            &wrap,
            top,
            LEAF_P1_CAP,
        )
    };
    assert!(
        verify(LeafQuery::V1, &top_lean).is_err(),
        "the lean tree under V1"
    );
    assert!(
        verify(LeafQuery::Lean, &top_v1).is_err(),
        "the V1 tree under lean"
    );
    verify(LeafQuery::V1, &top_v1).expect("the V1 tree under its own check");
    verify(LeafQuery::Lean, &top_lean).expect("the lean tree under its own check");
}

// ============================== the real block ============================

/// A node builder that stops leaves an error in every slot it had not
/// filled, so no level waits forever, and keeps what it published.
#[test]
fn a_stopped_node_builder_fails_every_empty_slot() {
    let slots: Vec<Vec<Published<u32>>> = vec![
        vec![Published::new(), Published::new()],
        vec![Published::new()],
    ];
    {
        let _fail = FailUnpublished(&slots);
        let _ = slots[0][1].0.set(Ok(7));
    }
    // `get`, not `wait`: an empty slot must fail the test, not hang it.
    assert_eq!(slots[0][1].0.get(), Some(&Ok(7)));
    for (lv, j) in [(0, 0), (1, 0)] {
        assert_eq!(
            slots[lv][j].0.get(),
            Some(&Err("the node builder stopped".to_string())),
            "slot ({lv}, {j})"
        );
    }
}

/// [`build_levels`] over a toy tree whose programs are their texts (`L3` a
/// leaf, `N(..)` / `T(..)` a node over its children's): each node's build
/// sleeps longer the earlier it is in its level, so on a pool the level's
/// nodes tend to finish in reverse. On a pool, each even node of level 0
/// finishes only after the odd node after it: the sleeps alone do not order
/// them, since a sleep can overrun the 4 ms between two neighbours (on a
/// loaded macOS host a 28 ms sleep took 38 ms, and the level then finished in
/// order). Returns every slot's content and the order the builds finished in.
#[allow(clippy::type_complexity)]
fn toy_tree(
    leaves: usize,
    arities: &[Vec<usize>],
    threads: Option<usize>,
    fail_at: Option<(usize, usize)>,
) -> (
    Vec<Vec<Option<Result<String, String>>>>,
    Vec<(usize, usize)>,
) {
    use super::per_table_aggregator::Level;
    let shape: Vec<Level> = arities
        .iter()
        .map(|a| Level { arities: a.clone() })
        .collect();
    let leaf_text: Vec<String> = (0..leaves).map(|k| format!("L{k}")).collect();
    let leaf_refs: Vec<&String> = leaf_text.iter().collect();
    let slots: Vec<Vec<Published<String>>> = shape
        .iter()
        .map(|l| l.arities.iter().map(|_| Published::new()).collect())
        .collect();
    let programs: Vec<Vec<Published<String>>> = shape
        .iter()
        .map(|l| l.arities.iter().map(|_| Published::new()).collect())
        .collect();
    let finished = std::sync::Mutex::new(Vec::new());
    // Level 0's builds that have finished, for the even nodes to wait on.
    let done = (
        std::sync::Mutex::new(vec![false; shape[0].arities.len()]),
        std::sync::Condvar::new(),
    );
    let pool = threads.map(|n| {
        rayon::ThreadPoolBuilder::new()
            .num_threads(n)
            .build()
            .expect("a pool")
    });
    build_levels(
        &shape,
        &|i: usize| -> Result<&String, String> { Ok(leaf_refs[i]) },
        leaf_refs.len(),
        &programs,
        &slots,
        |node: &String| node,
        |_, _| Ok(()),
        |lv, j, kids: &[&String], top, ()| {
            let late = shape[lv].arities.len() - j;
            std::thread::sleep(std::time::Duration::from_millis(4 * late as u64));
            if lv == 0 {
                let (marks, landed) = &done;
                let mut marks = marks.lock().expect("the marks");
                // Its odd neighbour is handed out next, to another thread, and
                // never waits: no two even nodes wait at once. Bounded, so a
                // neighbour that never comes fails the order check, not the
                // run.
                if threads.is_some() && j % 2 == 0 && j + 1 < marks.len() {
                    marks = landed
                        .wait_timeout_while(marks, std::time::Duration::from_secs(10), |m| {
                            !m[j + 1]
                        })
                        .expect("the marks")
                        .0;
                }
                finished.lock().expect("the log").push((lv, j));
                marks[j] = true;
                landed.notify_all();
            } else {
                finished.lock().expect("the log").push((lv, j));
            }
            if fail_at == Some((lv, j)) {
                return Err(format!("node ({lv}, {j}) does not build"));
            }
            let kids: Vec<&str> = kids.iter().map(|k| k.as_str()).collect();
            let text = format!("{}({})", if top { "T" } else { "N" }, kids.join(","));
            Ok((text.clone(), text))
        },
        // The finish is where a node takes the card: never on a rayon worker.
        // Its program is already published when it starts.
        |lv, j, program: String| match rayon::current_thread_index() {
            None if programs[lv][j].0.get() == Some(&Ok(program.clone())) => Ok(program),
            None => Err(format!(
                "node ({lv}, {j}) finished before its program was published"
            )),
            Some(w) => Err(format!("node ({lv}, {j}) finished on rayon worker {w}")),
        },
        pool.as_ref(),
    );
    // Every slot's program is published too, and agrees with its node (the
    // error included).
    for (lv, level) in slots.iter().enumerate() {
        for (j, slot) in level.iter().enumerate() {
            match (slot.0.get(), programs[lv][j].0.get()) {
                (Some(Ok(node)), program) => {
                    assert_eq!(program, Some(&Ok(node.clone())), "program ({lv}, {j})")
                }
                (_, program) => assert!(program.is_some(), "program ({lv}, {j}) never published"),
            }
        }
    }
    let got = slots
        .iter()
        .map(|level| level.iter().map(|s| s.0.get().cloned()).collect())
        .collect();
    (got, finished.into_inner().expect("the log"))
}

/// The median's tree shape: 23 leaves at fan-in 3.
fn median_arities() -> Vec<Vec<usize>> {
    vec![vec![3, 3, 3, 3, 3, 3, 3, 2], vec![3, 3, 2], vec![3]]
}

/// The toy tree's nodes over `leaves` leaves, level by level, as [`toy_tree`]
/// names them.
fn toy_want(leaves: usize, arities: &[Vec<usize>]) -> Vec<Vec<Option<Result<String, String>>>> {
    let group =
        |kids: &[String], top: bool| format!("{}({})", if top { "T" } else { "N" }, kids.join(","));
    let mut want: Vec<Vec<String>> = Vec::new();
    let mut below: Vec<String> = (0..leaves).map(|k| format!("L{k}")).collect();
    for (lv, a) in arities.iter().enumerate() {
        let top = lv + 1 == arities.len();
        let mut at = 0;
        let level: Vec<String> = a
            .iter()
            .map(|&n| {
                at += n;
                group(&below[at - n..at], top)
            })
            .collect();
        want.push(level.clone());
        below = level;
    }
    want.into_iter()
        .map(|l| l.into_iter().map(|p| Some(Ok(p))).collect())
        .collect()
}

/// ★ The node pipe builds the serial builder's nodes, each in its own slot,
/// whatever order a level's emissions finish in, and finishes every node (where
/// it takes the card) off the rayon workers: the median's shape (23 leaves at
/// fan-in 3), on pools of 2 to 4 threads, where each level's emissions finish
/// out of order, and serially.
#[test]
fn the_node_pipe_builds_the_serial_builders_nodes_in_any_completion_order() {
    let arities = median_arities();
    let want = toy_want(23, &arities);
    for threads in [None, Some(2), Some(3), Some(4)] {
        let (got, finished) = toy_tree(23, &arities, threads, None);
        assert_eq!(got, want, "{threads:?} threads");
        let level0: Vec<usize> = finished
            .iter()
            .filter(|(lv, _)| *lv == 0)
            .map(|(_, j)| *j)
            .collect();
        let in_order = level0.windows(2).all(|w| w[0] < w[1]);
        assert_eq!(
            in_order,
            threads.is_none(),
            "{threads:?} threads: level 0 finished in {level0:?}"
        );
    }
}

/// A node build that fails leaves its error in its slot and an error in every
/// slot above it, promptly: no level waits for a node that will never come.
#[test]
fn a_failing_node_build_fails_every_node_above_it() {
    let (tx, rx) = std::sync::mpsc::channel();
    std::thread::spawn(move || {
        let arities = vec![vec![3, 3, 3, 3, 3, 3, 3, 2], vec![3, 3, 2], vec![3]];
        let _ = tx.send(toy_tree(23, &arities, Some(3), Some((0, 2))).0);
    });
    let got = rx
        .recv_timeout(std::time::Duration::from_secs(20))
        .expect("the builder hung after a failed build");
    assert_eq!(
        got[0][2],
        Some(Err("node (0, 2) does not build".to_string()))
    );
    for (lv, level) in got.iter().enumerate().skip(1) {
        for (j, slot) in level.iter().enumerate() {
            assert!(
                matches!(slot, Some(Err(_))),
                "slot ({lv}, {j}) above a failed build: {slot:?}"
            );
        }
    }
}

/// [`build_levels`] at the median's shape over toy leaves (`L3`) that
/// [`publish_each`] builds one after another on another thread, each waited
/// for where it is read, as [`build_nodes`] waits for a leaf's artifacts. While
/// it builds the last leaf the publisher waits up to `poll` for the first
/// node's program; the leaf at `fail_leaf` does not build. Returns every node
/// slot's content and whether that program was out before the last leaf was
/// built — or `None` if the two did not finish within 20 s.
#[allow(clippy::type_complexity)]
fn toy_tree_over_published_leaves(
    threads: Option<usize>,
    together: bool,
    fail_leaf: Option<usize>,
    poll: std::time::Duration,
) -> Option<(Vec<Vec<Option<Result<String, String>>>>, bool)> {
    let (tx, rx) = std::sync::mpsc::channel();
    std::thread::spawn(move || {
        use super::per_table_aggregator::Level;
        let n = 23;
        let shape: Vec<Level> = median_arities()
            .into_iter()
            .map(|arities| Level { arities })
            .collect();
        let node_slots = || -> Vec<Vec<Published<String>>> {
            shape
                .iter()
                .map(|l| l.arities.iter().map(|_| Published::new()).collect())
                .collect()
        };
        let (programs, slots) = (node_slots(), node_slots());
        let leaves: Vec<Published<String>> = (0..n).map(|_| Published::new()).collect();
        let pool = threads.map(|t| {
            rayon::ThreadPoolBuilder::new()
                .num_threads(t)
                .build()
                .expect("a pool")
        });
        let early = std::sync::atomic::AtomicBool::new(false);
        std::thread::scope(|scope| {
            scope.spawn(|| {
                let ids: Vec<usize> = (0..n).collect();
                publish_each(
                    &ids,
                    &leaves,
                    !together,
                    |_| true,
                    |k, _| {
                        std::thread::sleep(std::time::Duration::from_millis(2));
                        if fail_leaf == Some(k) {
                            return Err(format!("leaf {k} does not build"));
                        }
                        if k + 1 == n {
                            let t = std::time::Instant::now();
                            while programs[0][0].0.get().is_none() && t.elapsed() < poll {
                                std::thread::sleep(std::time::Duration::from_millis(1));
                            }
                            let out = programs[0][0].0.get().is_some();
                            early.store(out, std::sync::atomic::Ordering::SeqCst);
                        }
                        Ok(format!("L{k}"))
                    },
                )
            });
            build_levels(
                &shape,
                &|i: usize| leaves[i].wait(),
                n,
                &programs,
                &slots,
                |node: &String| node,
                |_, _| Ok(()),
                |_, _, kids: &[&String], top, ()| {
                    let kids: Vec<&str> = kids.iter().map(|k| k.as_str()).collect();
                    let text = format!("{}({})", if top { "T" } else { "N" }, kids.join(","));
                    Ok((text.clone(), text))
                },
                |_, _, program: String| Ok(program),
                pool.as_ref(),
            );
        });
        let got: Vec<Vec<Option<Result<String, String>>>> = slots
            .iter()
            .map(|level| level.iter().map(|s| s.0.get().cloned()).collect())
            .collect();
        let _ = tx.send((got, early.into_inner()));
    });
    rx.recv_timeout(std::time::Duration::from_secs(20)).ok()
}

/// ★ Leaves published one at a time build the tree all of them published at
/// once build, and nothing waits forever: serially and on pools of 1 to 4
/// threads, where a node over leaves waits for its own on a pool worker.
#[test]
fn leaves_published_one_at_a_time_build_the_same_tree() {
    let want = toy_want(23, &median_arities());
    for threads in [None, Some(1), Some(2), Some(4)] {
        for together in [false, true] {
            let got =
                toy_tree_over_published_leaves(threads, together, None, std::time::Duration::ZERO);
            let (got, _) = got.unwrap_or_else(|| {
                panic!("{threads:?} threads, together {together}: the tree hung")
            });
            assert_eq!(got, want, "{threads:?} threads, together {together}");
        }
    }
}

/// ★ A node over leaves is emitted once its own leaves are published, before
/// the last leaf is built; with every leaf held back to the end (the default
/// while every leaf's program is there), it cannot be.
#[test]
fn a_node_over_leaves_is_emitted_before_the_last_leaf_is_built() {
    let want = toy_want(23, &median_arities());
    for threads in [None, Some(2), Some(4)] {
        for (together, poll) in [(false, 10_000), (true, 200)] {
            let poll = std::time::Duration::from_millis(poll);
            let (got, early) = toy_tree_over_published_leaves(threads, together, None, poll)
                .unwrap_or_else(|| {
                    panic!("{threads:?} threads, together {together}: the tree hung")
                });
            assert_eq!(got, want, "{threads:?} threads, together {together}");
            assert_eq!(
                early, !together,
                "{threads:?} threads, together {together}: the first node's program \
                 out before the last leaf was built"
            );
        }
    }
}

/// A leaf that does not build fails the nodes over it and every node above,
/// and the leaves after it fail too: nothing waits forever.
#[test]
fn a_leaf_that_does_not_build_fails_the_nodes_over_it() {
    for threads in [None, Some(4)] {
        for together in [false, true] {
            let (got, _) = toy_tree_over_published_leaves(
                threads,
                together,
                Some(4),
                std::time::Duration::ZERO,
            )
            .unwrap_or_else(|| panic!("{threads:?} threads, together {together}: the tree hung"));
            let at = format!("{threads:?} threads, together {together}");
            assert_eq!(
                got[0][0],
                Some(Ok("N(L0,L1,L2)".to_string())),
                "{at}: the node before it"
            );
            assert_eq!(
                got[0][1],
                Some(Err("leaf 4 does not build".to_string())),
                "{at}: the node over it"
            );
            for (lv, level) in got.iter().enumerate() {
                for (j, slot) in level.iter().enumerate().skip(usize::from(lv == 0) * 2) {
                    assert!(
                        matches!(slot, Some(Err(_))),
                        "{at}: slot ({lv}, {j}) after a failed leaf: {slot:?}"
                    );
                }
            }
        }
    }
}

/// ★ [`publish_each`] holds what it built while each next input is there,
/// publishing it all after the last; before an input that is not there yet it
/// publishes what it holds, then each value as it is built. Here input 3
/// arrives only once output 0 is out, as a leaf program the budget admits only
/// once the provers take the leaves before it: holding across it (the mutation:
/// ignore `ready`) waits forever, and the bounded wait fails the test instead.
#[test]
fn the_leaves_artifacts_are_published_before_a_wait_for_a_missing_program() {
    let wait_for = |slot: &Published<usize>| -> Result<usize, String> {
        let t = std::time::Instant::now();
        while slot.0.get().is_none() {
            if t.elapsed() > std::time::Duration::from_secs(10) {
                return Err("an input that never came".to_string());
            }
            std::thread::sleep(std::time::Duration::from_millis(1));
        }
        slot.wait().copied()
    };
    // Every input there: one publication, after the last.
    let inputs: Vec<Published<usize>> = (0..6).map(|_| Published::new()).collect();
    for (k, input) in inputs.iter().enumerate() {
        let _ = input.0.set(Ok(k));
    }
    let out: Vec<Published<usize>> = (0..6).map(|_| Published::new()).collect();
    let from = publish_each(
        &inputs,
        &out,
        false,
        |k| inputs[k].0.get().is_some(),
        |k, input| {
            assert!(
                out[0].0.get().is_none(),
                "output 0 out before the last build ({k})"
            );
            wait_for(input).map(|v| v * 10)
        },
    );
    assert_eq!(from, None);
    let got: Vec<_> = out.iter().map(|s| s.0.get().cloned()).collect();
    assert_eq!(got, (0..6).map(|k| Some(Ok(k * 10))).collect::<Vec<_>>());
    // Input 3 comes only once output 0 is out.
    let inputs: Vec<Published<usize>> = (0..6).map(|_| Published::new()).collect();
    for (k, input) in inputs.iter().enumerate().take(3) {
        let _ = input.0.set(Ok(k));
    }
    let out: Vec<Published<usize>> = (0..6).map(|_| Published::new()).collect();
    let from = std::thread::scope(|scope| {
        scope.spawn(|| {
            if wait_for(&out[0]).is_ok() {
                for (k, input) in inputs.iter().enumerate().skip(3) {
                    let _ = input.0.set(Ok(k));
                }
            }
        });
        publish_each(
            &inputs,
            &out,
            false,
            |k| inputs[k].0.get().is_some(),
            |_, input| wait_for(input).map(|v| v * 10),
        )
    });
    assert_eq!(
        from,
        Some(3),
        "per-item publication begins at the missing input"
    );
    let got: Vec<_> = out.iter().map(|s| s.0.get().cloned()).collect();
    assert_eq!(got, (0..6).map(|k| Some(Ok(k * 10))).collect::<Vec<_>>());
}

/// The W3 tree's programs composed as `prove_tree_pipelined` composes them,
/// over toy programs (texts), at the median's shape: under `room`, the leaves
/// the budget admits now are emitted at once and the rest by the streaming
/// emitter as it admits them; the leaves' "artifacts" (their texts) published
/// by [`publish_each`] against "is the next program there"; the nodes admitted
/// in prove order on the builder's plain threads ([`build_levels`]); the
/// provers taking programs in tree order, claiming each at the take and
/// letting it go after its proof ([`prove_dataflow`], [`node_flow`]). Returns
/// every node's text, how many leaves came in at once, where per-leaf
/// publication began, the budget's summary and how many programs were never
/// let go — or `None` after 30 s.
#[allow(clippy::type_complexity)]
fn toy_budget_tree(
    threads: Option<usize>,
    room: super::program_budget::Room,
    ahead: usize,
) -> Option<(
    Vec<Vec<Option<Result<String, String>>>>,
    usize,
    Option<usize>,
    String,
    usize,
)> {
    use super::per_table_aggregator::Level;
    use super::program_budget::{Permit, ProgramBudget, emit_ordered};
    let (tx, rx) = std::sync::mpsc::channel();
    std::thread::spawn(move || {
        let n = 23;
        let shape: Vec<Level> = median_arities()
            .into_iter()
            .map(|arities| Level { arities })
            .collect();
        let budget = ProgramBudget::with_reading(room, ahead, Box::new(|| None));
        let tb = TreeBudget::new(budget, &vec![1; n], &shape);
        // Beside the base: the leaves admitted now, in order.
        let permits: Vec<Permit> = (0..n)
            .map_while(|k| tb.budget.try_acquire(k, tb.leaf_estimate(k)))
            .collect();
        let upto = permits.len();
        let leaves: Vec<Published<ProgramSlot<String>>> =
            (0..n).map(|_| Published::new()).collect();
        for (k, mut permit) in permits.into_iter().enumerate() {
            permit.emitted(1);
            let _ = leaves[k]
                .0
                .set(Ok(ProgramSlot::new(format!("L{k}"), Some(permit))));
        }
        let built: Vec<Published<String>> = (0..n).map(|_| Published::new()).collect();
        let node_slots = || -> Vec<Vec<Published<String>>> {
            shape
                .iter()
                .map(|l| l.arities.iter().map(|_| Published::new()).collect())
                .collect()
        };
        let slots = node_slots();
        let programs: Vec<Vec<Published<ProgramSlot<String>>>> = shape
            .iter()
            .map(|l| l.arities.iter().map(|_| Published::new()).collect())
            .collect();
        let results: Vec<Vec<Published<String>>> = std::iter::once(n)
            .chain(shape.iter().map(|l| l.arities.len()))
            .map(|m| (0..m).map(|_| Published::new()).collect())
            .collect();
        let pool = threads.map(|t| {
            rayon::ThreadPoolBuilder::new()
                .num_threads(t)
                .build()
                .expect("a pool")
        });
        let tb = &tb;
        let from = std::thread::scope(|outer| {
            if upto < n {
                let leaves = &leaves;
                outer.spawn(move || {
                    let _fail = FailEmpty(&leaves[upto..], "the leaf emitter stopped");
                    let emit = |k: usize| -> Result<ProgramSlot<String>, String> {
                        let mut permit = tb.budget.acquire(k, tb.leaf_estimate(k))?;
                        permit.emitted(1);
                        Ok(ProgramSlot::new(format!("L{k}"), Some(permit)))
                    };
                    emit_ordered(upto..n, 4, &emit, &mut |k, slot| {
                        let emitted = slot.is_ok();
                        let _ = leaves[k].0.set(slot);
                        emitted
                    });
                });
            }
            outer.spawn(|| {
                build_levels(
                    &shape,
                    &|i: usize| built[i].wait(),
                    n,
                    &programs,
                    &slots,
                    |node: &String| node,
                    |lv, j| {
                        tb.budget
                            .acquire(tb.node_order(lv, j), tb.node_estimate())
                            .map(Some)
                    },
                    |_, _, kids: &[&String], top, permit: Option<Permit>| {
                        let kids: Vec<&str> = kids.iter().map(|k| k.as_str()).collect();
                        let text = format!("{}({})", if top { "T" } else { "N" }, kids.join(","));
                        let mut permit = permit;
                        if let Some(p) = permit.as_mut() {
                            p.emitted(1);
                        }
                        Ok((ProgramSlot::new(text.clone(), permit), text))
                    },
                    |_, _, text: String| Ok(text),
                    pool.as_ref(),
                )
            });
            std::thread::scope(|scope| {
                let artifacts = scope.spawn(|| {
                    publish_each(
                        &leaves,
                        &built,
                        false,
                        |k| leaves[k].0.get().is_some(),
                        |_, slot| Ok(slot.wait()?.get()?.program.clone()),
                    )
                });
                prove_dataflow(
                    &shape,
                    &tree_order(n, &shape),
                    &results,
                    3,
                    false,
                    |at, kids: &[&Published<String>]| -> Result<String, String> {
                        if at.lv == 0 {
                            let slot = leaves[at.j].wait()?;
                            slot.claim();
                            let program = slot.get()?.program.clone();
                            built[at.j].wait()?;
                            std::thread::sleep(std::time::Duration::from_millis(1));
                            slot.release();
                            return Ok(program);
                        }
                        node_flow(
                            &programs,
                            &slots,
                            at,
                            true,
                            |slot: &ProgramSlot<String>| -> Result<String, String> {
                                slot.claim();
                                let program = slot.get()?.program.clone();
                                wait_all(kids)?;
                                slot.release();
                                Ok(program)
                            },
                            |program, _: &String| Ok(program),
                        )
                    },
                );
                tb.budget.fail();
                artifacts.join().expect("the artifacts thread")
            })
        });
        let got: Vec<Vec<Option<Result<String, String>>>> = slots
            .iter()
            .map(|level| level.iter().map(|s| s.0.get().cloned()).collect())
            .collect();
        let kept = leaves
            .iter()
            .chain(programs.iter().flatten())
            .filter(|slot| slot.wait().is_ok_and(|p| p.get().is_ok()))
            .count();
        let _ = tx.send((got, upto, from, tb.budget.summary(), kept));
    });
    rx.recv_timeout(std::time::Duration::from_secs(30)).ok()
}

/// ★ No room at all (a one-byte budget; every program comes in among the next
/// `ahead` the provers will take): the tree still completes with the same
/// nodes, serially and on pools of 1 to 4 threads, ahead 1 and
/// [`super::program_budget::AHEAD`]. The leaves' artifacts go out before the
/// first missing program; holding them across it (or provers that never claim)
/// deadlocks, which the bounded run turns into a failure. With room, every leaf
/// comes in at once and the artifacts go out together, as before the budget.
#[test]
fn a_one_byte_budget_cannot_deadlock_the_w3_tree() {
    use super::program_budget::{AHEAD, Room};
    let want = toy_want(23, &median_arities());
    for threads in [None, Some(1), Some(2), Some(4)] {
        for ahead in [1, AHEAD] {
            let (got, upto, from, summary, kept) = toy_budget_tree(threads, Room::Bytes(1), ahead)
                .unwrap_or_else(|| panic!("{threads:?} threads, ahead {ahead}: the tree hung"));
            assert_eq!(got, want, "{threads:?} threads, ahead {ahead}");
            assert_eq!(kept, 0, "{threads:?} threads: programs never let go");
            assert_eq!(upto, ahead, "{threads:?} threads: leaves in at once");
            // The provers claim the leaves they take before those leaves'
            // artifacts are out, so a few more programs may come in first; at
            // the latest, the leaf after the three in flight is missing.
            assert!(
                from.is_some_and(|f| (ahead..=ahead + 3).contains(&f)),
                "{threads:?} threads, ahead {ahead}: per-leaf from {from:?}"
            );
            assert!(
                summary.contains(&format!("0 by room, 35 among the next {ahead}")),
                "{threads:?} threads, ahead {ahead}: {summary}"
            );
        }
    }
    let (got, upto, from, summary, kept) =
        toy_budget_tree(Some(4), Room::Unbounded, AHEAD).expect("the roomy tree hung");
    assert_eq!(got, want);
    assert_eq!(kept, 0, "programs never let go");
    assert_eq!(
        (upto, from),
        (23, None),
        "every leaf at once, published together"
    );
    assert!(summary.contains("35 by room, 0 among"), "{summary}");
}

/// One node's stamps in [`split_slot_flow`], in ms since the tree started:
/// when a worker took it, when its program was emitted, when the worker got
/// the program, when it executed and proved; and whose artifacts it proved
/// with, and when those were built.
#[derive(Clone, Debug)]
struct FlowStamps {
    taken: u128,
    emitted: u128,
    got: u128,
    exec: u128,
    prove: u128,
    artifacts: String,
    ready: u128,
}

/// The tree's split slots over a toy at the median's shape (23 leaves, fan-in
/// 3): [`build_levels`] emits each level's programs on a pool in reverse (the
/// last node first), publishes each program and then each node's artifacts
/// after a finish that holds "the card" — 120 ms for the first node to arrive,
/// 5 ms for the rest; [`prove_dataflow`] proves the leaves and runs each node
/// through [`node_flow`]. Returns every node's stamps, by level then index.
fn split_slot_flow(workers: usize) -> Vec<Vec<FlowStamps>> {
    use super::per_table_aggregator::Level;
    let shape: Vec<Level> = [vec![3, 3, 3, 3, 3, 3, 3, 2], vec![3, 3, 2], vec![3]]
        .into_iter()
        .map(|arities| Level { arities })
        .collect();
    let leaf_text: Vec<String> = (0..23).map(|k| format!("L{k}")).collect();
    let leaf_refs: Vec<&String> = leaf_text.iter().collect();
    let node_slots = || -> Vec<Vec<Published<(String, u128)>>> {
        shape
            .iter()
            .map(|l| l.arities.iter().map(|_| Published::new()).collect())
            .collect()
    };
    // A program is (its text, when emitted); a node is (its artifacts' name,
    // when built).
    let (programs, slots) = (node_slots(), node_slots());
    let results: Vec<Vec<Published<FlowStamps>>> = std::iter::once(23)
        .chain(shape.iter().map(|l| l.arities.len()))
        .map(|n| (0..n).map(|_| Published::new()).collect())
        .collect();
    let t0 = std::time::Instant::now();
    let ms = || t0.elapsed().as_millis();
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(4)
        .build()
        .expect("a pool");
    std::thread::scope(|scope| {
        scope.spawn(|| {
            build_levels(
                &shape,
                &|i: usize| -> Result<&String, String> { Ok(leaf_refs[i]) },
                leaf_refs.len(),
                &programs,
                &slots,
                |node: &(String, u128)| &node.0,
                |_, _| Ok(()),
                |lv, j, kids: &[&String], _, ()| {
                    let late = shape[lv].arities.len() - j;
                    std::thread::sleep(std::time::Duration::from_millis(3 * late as u64));
                    let text = format!(
                        "N({})",
                        kids.iter()
                            .map(|k| k.as_str())
                            .collect::<Vec<_>>()
                            .join(",")
                    );
                    Ok(((text.clone(), ms()), text))
                },
                |lv, j, _: String| {
                    let first = j + 1 == shape[lv].arities.len();
                    let hold = if first { 120 } else { 5 };
                    std::thread::sleep(std::time::Duration::from_millis(hold));
                    Ok((format!("A{lv}.{j}"), ms()))
                },
                Some(&pool),
            )
        });
        prove_dataflow(
            &shape,
            &tree_order(23, &shape),
            &results,
            workers,
            false,
            |at, kids: &[&Published<FlowStamps>]| {
                let taken = ms();
                if at.lv == 0 {
                    std::thread::sleep(std::time::Duration::from_millis(1));
                    let now = ms();
                    return Ok(FlowStamps {
                        taken,
                        emitted: 0,
                        got: taken,
                        exec: now,
                        prove: now,
                        artifacts: String::new(),
                        ready: 0,
                    });
                }
                node_flow(
                    &programs,
                    &slots,
                    at,
                    true,
                    |program: &(String, u128)| {
                        let got = ms();
                        wait_all(kids)?;
                        let exec = ms();
                        std::thread::sleep(std::time::Duration::from_millis(1));
                        Ok((program.1, got, exec))
                    },
                    |(emitted, got, exec), node: &(String, u128)| {
                        Ok(FlowStamps {
                            taken,
                            emitted,
                            got,
                            exec,
                            prove: ms(),
                            artifacts: node.0.clone(),
                            ready: node.1,
                        })
                    },
                )
            },
        );
    });
    results
        .into_iter()
        .skip(1)
        .map(|level| {
            level
                .into_iter()
                .map(|s| s.take().expect("every node proves"))
                .collect()
        })
        .collect()
}

/// ★ A node gets its program as soon as it is emitted — not behind another
/// node's finish holding the card — and executes from it before its artifacts
/// are built, but proves only with its OWN artifacts, after they are built: at
/// 1 to 6 workers, without a hang. (Publishing a program after a sibling's
/// finish, waiting on another node's artifact slot, or waiting for the
/// artifacts before the execute each fails this.)
#[test]
fn a_node_proves_only_after_its_own_artifacts_and_executes_before_them() {
    for workers in 1..=6 {
        let got = within_20s(move || split_slot_flow(workers));
        let mut early = 0usize;
        for (l, level) in got.iter().enumerate() {
            for (j, node) in level.iter().enumerate() {
                let at = format!("{workers} workers: node ({}, {j})", l + 1);
                assert_eq!(node.artifacts, format!("A{l}.{j}"), "{at}");
                assert!(
                    node.prove >= node.ready,
                    "{at} proved at {} ms, before its artifacts at {} ms",
                    node.prove,
                    node.ready
                );
                assert!(
                    node.got <= node.taken.max(node.emitted) + 40,
                    "{at} got its program at {} ms: emitted at {} ms, taken at {} ms",
                    node.got,
                    node.emitted,
                    node.taken
                );
                early += usize::from(node.exec < node.ready);
            }
        }
        assert!(
            early > 0,
            "{workers} workers: no node executed before its artifacts were built"
        );
    }
}

/// The top streams when its program is ready before its last child is proved
/// (as at the median), and executes whole when it is ready after (as at 1×).
#[test]
fn the_top_streams_only_while_a_child_is_unproved() {
    let kids: Vec<Published<u32>> = (0..3).map(|_| Published::new()).collect();
    let refs: Vec<&Published<u32>> = kids.iter().collect();
    assert!(top_streams(&refs), "ready before any child: streamed");
    let _ = kids[2].0.set(Ok(20));
    let _ = kids[0].0.set(Ok(0));
    assert!(top_streams(&refs), "ready before the last child: streamed");
    let _ = kids[1].0.set(Ok(10));
    assert!(!top_streams(&refs), "ready after the last child: whole");

    let kids: Vec<Published<u32>> = (0..2).map(|_| Published::new()).collect();
    let _ = kids[0].0.set(Ok(0));
    let _ = kids[1].0.set(Err("child 1 failed".to_string()));
    let refs: Vec<&Published<u32>> = kids.iter().collect();
    assert!(!top_streams(&refs), "a failed child is done: whole");
    assert_eq!(
        wait_all(&refs).map(|_| ()),
        Err("child 1 failed".to_string()),
        "and the whole path reports the failure"
    );
}

/// The streamed top's children land in the order they are proved, each once,
/// and a failed child fails the top with its error, without a hang.
#[test]
fn a_streamed_top_takes_its_children_as_they_are_proved() {
    let kids: Vec<Published<u32>> = (0..3).map(|_| Published::new()).collect();
    let order = std::thread::scope(|scope| {
        let kids = &kids;
        scope.spawn(move || {
            for c in [2usize, 0, 1] {
                std::thread::sleep(std::time::Duration::from_millis(15));
                let _ = kids[c].0.set(Ok(10 * c as u32));
            }
        });
        let refs: Vec<&Published<u32>> = kids.iter().collect();
        let mut order = Vec::new();
        in_arrival_order(&refs, |c, v| {
            assert_eq!(*v, 10 * c as u32);
            order.push(c);
            Ok(())
        })
        .expect("every child lands");
        order
    });
    assert_eq!(order, vec![2, 0, 1]);

    let kids: Vec<Published<u32>> = (0..3).map(|_| Published::new()).collect();
    let _ = kids[1].0.set(Ok(10));
    let _ = kids[0].0.set(Err("child 0 failed".to_string()));
    let refs: Vec<&Published<u32>> = kids.iter().collect();
    // Child 0's error is already published, so this returns at once.
    let got = in_arrival_order(&refs, |_, _| Ok(()));
    assert_eq!(got, Err("child 0 failed".to_string()));
}

/// [`prove_dataflow`] over a toy tree at the median's shape (23 leaves, fan-in
/// 3, levels of 8, 3 and 1) whose programs are their texts (`L3` a leaf,
/// `N(..)` / `T(..)` a node over its children's): every program sleeps longer
/// the earlier it is in its level, so a level's programs finish in reverse.
/// Returns each slot's text (or error) and each program's (start, end), in
/// milliseconds since the tree started.
#[allow(clippy::type_complexity)]
fn toy_flow(
    workers: usize,
    barrier: bool,
    fail_at: Option<TreeAt>,
    panic_at: Option<TreeAt>,
    order: impl Fn(&[TreeAt]) -> Vec<TreeAt>,
) -> Vec<Vec<Result<(String, u128, u128), String>>> {
    use super::per_table_aggregator::Level;
    let shape: Vec<Level> = [vec![3, 3, 3, 3, 3, 3, 3, 2], vec![3, 3, 2], vec![3]]
        .into_iter()
        .map(|arities| Level { arities })
        .collect();
    let results: Vec<Vec<Published<(String, u128, u128)>>> = std::iter::once(23)
        .chain(shape.iter().map(|l| l.arities.len()))
        .map(|n| (0..n).map(|_| Published::new()).collect())
        .collect();
    let width = |lv: usize| {
        if lv == 0 {
            23
        } else {
            shape[lv - 1].arities.len()
        }
    };
    let t0 = std::time::Instant::now();
    prove_dataflow(
        &shape,
        &order(&tree_order(23, &shape)),
        &results,
        workers,
        barrier,
        |at, kids: &[&Published<(String, u128, u128)>]| {
            let kids = wait_all(kids)?;
            let start = t0.elapsed().as_millis();
            let late = width(at.lv) - at.j;
            std::thread::sleep(std::time::Duration::from_millis(3 * late as u64));
            if fail_at == Some(at) {
                return Err(format!("({}, {}) does not prove", at.lv, at.j));
            }
            assert!(panic_at != Some(at), "({}, {}) panics", at.lv, at.j);
            let text = match at.lv {
                0 => format!("L{}", at.j),
                _ => {
                    let kids: Vec<&str> = kids.iter().map(|k| k.0.as_str()).collect();
                    let top = at.lv == shape.len();
                    format!("{}({})", if top { "T" } else { "N" }, kids.join(","))
                }
            };
            Ok((text, start, t0.elapsed().as_millis()))
        },
    );
    results
        .iter()
        .map(|level| {
            level
                .iter()
                .map(|s| s.0.get().cloned().unwrap_or(Err("empty".to_string())))
                .collect()
        })
        .collect()
}

/// Runs `f` on a thread of its own and gives up after 20 s: a deadlocked
/// scheduler fails the test instead of hanging it.
fn within_20s<T: Send + 'static>(f: impl FnOnce() -> T + Send + 'static) -> T {
    let (tx, rx) = std::sync::mpsc::channel();
    std::thread::spawn(move || {
        let _ = tx.send(f());
    });
    rx.recv_timeout(std::time::Duration::from_secs(20))
        .expect("the tree's scheduler deadlocked (no result in 20 s)")
}

/// [`prove_dataflow`] over the toy's shape where the last leaf, once taken,
/// holds its worker until some node starts (or 1 s passes): whether a node
/// started while that leaf was still proving. No clock decides it, only who
/// signalled whom.
fn a_node_starts_while_the_last_leaf_proves(workers: usize, barrier: bool) -> bool {
    use super::per_table_aggregator::Level;
    let shape: Vec<Level> = [vec![3, 3, 3, 3, 3, 3, 3, 2], vec![3, 3, 2], vec![3]]
        .into_iter()
        .map(|arities| Level { arities })
        .collect();
    let results: Vec<Vec<Published<()>>> = std::iter::once(23)
        .chain(shape.iter().map(|l| l.arities.len()))
        .map(|n| (0..n).map(|_| Published::new()).collect())
        .collect();
    let (started_tx, started_rx) = std::sync::mpsc::channel::<()>();
    let (started_tx, started_rx) = (
        std::sync::Mutex::new(started_tx),
        std::sync::Mutex::new(started_rx),
    );
    let overlapped = std::sync::atomic::AtomicBool::new(false);
    prove_dataflow(
        &shape,
        &tree_order(23, &shape),
        &results,
        workers,
        barrier,
        |at, kids: &[&Published<()>]| {
            wait_all(kids)?;
            match (at.lv, at.j) {
                (0, 22) => {
                    let rx = started_rx.lock().expect("one receiver");
                    if rx.recv_timeout(std::time::Duration::from_secs(1)).is_ok() {
                        overlapped.store(true, std::sync::atomic::Ordering::SeqCst);
                    }
                }
                (0, _) => {}
                _ => {
                    let _ = started_tx.lock().expect("never poisoned").send(());
                }
            }
            Ok(())
        },
    );
    overlapped.load(std::sync::atomic::Ordering::SeqCst)
}

/// ★ Dataflow over the median's shape proves every program of the serial
/// level-by-level order, each in its own slot, whatever order they finish in
/// (each level in reverse), at 1 to 4 workers: and with two or more workers a
/// node proves while the level below is still proving, which the barrier
/// (the control) and a single worker never let it. The overlap is decided by
/// signals, not by timestamps: the last leaf waits for a node to start.
#[test]
fn dataflow_proves_the_level_by_level_programs_in_any_completion_order() {
    let leaves: Vec<String> = (0..23).map(|k| format!("L{k}")).collect();
    let mut want = vec![leaves];
    for (lv, arities) in [vec![3, 3, 3, 3, 3, 3, 3, 2], vec![3, 3, 2], vec![3]]
        .iter()
        .enumerate()
    {
        let top = lv == 2;
        let below = want.last().expect("a level").clone();
        let mut at = 0;
        let level: Vec<String> = arities
            .iter()
            .map(|&n| {
                at += n;
                let kids = below[at - n..at].join(",");
                format!("{}({kids})", if top { "T" } else { "N" })
            })
            .collect();
        want.push(level);
    }
    for workers in 1..=4 {
        for barrier in [false, true] {
            let got = within_20s(move || toy_flow(workers, barrier, None, None, |o| o.to_vec()));
            let texts: Vec<Vec<String>> = got
                .iter()
                .map(|l| {
                    l.iter()
                        .map(|r| r.as_ref().expect("proved").0.clone())
                        .collect()
                })
                .collect();
            assert_eq!(texts, want, "{workers} workers, barrier {barrier}");
            // Does a node start while the level below is still proving?
            let early =
                within_20s(move || a_node_starts_while_the_last_leaf_proves(workers, barrier));
            assert_eq!(
                early,
                workers >= 2 && !barrier,
                "{workers} workers, barrier {barrier}: a node started before its level below ended"
            );
        }
    }
}

/// The no-deadlock invariant: workers take the programs in topological order,
/// so the earliest unfinished program never waits on an untaken one — at any
/// number of workers, with every level finishing in reverse. (Taking the nodes
/// first deadlocks this test: every worker waits on children no one has taken.)
#[test]
fn dataflow_in_topological_order_never_deadlocks() {
    for workers in 1..=6 {
        let got = within_20s(move || toy_flow(workers, false, None, None, |o| o.to_vec()));
        assert!(got.iter().flatten().all(Result::is_ok), "{workers} workers");
    }
}

/// A failed or panicking prove leaves its error in its slot and an error in
/// every slot above it, promptly: no node waits for a child that will never
/// come.
#[test]
fn a_failed_program_fails_every_program_above_it_without_a_hang() {
    for (fail, panic) in [
        (Some(TreeAt { lv: 0, j: 4 }), None),
        (None, Some(TreeAt { lv: 1, j: 2 })),
    ] {
        let got = within_20s(move || toy_flow(3, false, fail, panic, |o| o.to_vec()));
        let at = fail.or(panic).expect("one");
        assert!(got[at.lv][at.j].is_err(), "{at:?}: {:?}", got[at.lv][at.j]);
        let mut j = at.j;
        let arities = [vec![3, 3, 3, 3, 3, 3, 3, 2], vec![3, 3, 2], vec![3]];
        for lv in at.lv + 1..got.len() {
            let mut start = 0;
            j = arities[lv - 1]
                .iter()
                .position(|&n| {
                    start += n;
                    j < start
                })
                .expect("a parent");
            assert!(
                got[lv][j].is_err(),
                "{at:?}: its ancestor ({lv}, {j}) proved"
            );
        }
        assert!(
            got[0]
                .iter()
                .enumerate()
                .all(|(k, r)| r.is_ok() || k == at.j),
            "{at:?}: another leaf failed"
        );
    }
}

/// ★ W3's readout on a real block (box only, `--ignored`, `--features cuda`;
/// the block path reads no WHIR hash knob), through the one driver the CLI's
/// `prove-block`
/// runs too ([`super::whir_block_tree::prove_whir_block_tree`]), run the way a
/// prover would:
/// - the block proved in one proof at the production format;
/// - the moment its statement is final (after phase A), the plan derived and
///   the leaves' programs emitted on threads of their own, beside phase B;
/// - then the tree ([`prove_tree_pipelined`]): the leaves `W3_SIBLINGS` at a
///   time (3 by default), the nodes' programs built beside them.
///
/// Off the clock, after the whole block: every proof verified against its
/// artifacts, the base verified on the host, and the top checked by the block's
/// verifier against the top program it derives from the ELF and the statement.
/// Reads `BLOCK_WHIR_ELF`, `BLOCK_WHIR_INPUT`, `W3_LEAVES` (the rule's count
/// when unset), `W3_SIBLINGS`, `W3_FAN_IN` ([`BLOCK_FAN_IN`] when unset) and
/// `W3_CROSS_SCALE` (unset: no cross-scale verify).
/// Prints `W3 …` lines.
#[test]
#[ignore = "a real block and its recursion: box only"]
fn the_whir_block_tree_on_a_real_block() {
    use super::whir_block_tree::{StdoutSink, WhirTreeConfig, prove_whir_block_tree};
    let elf = std::fs::read(std::env::var("BLOCK_WHIR_ELF").expect("BLOCK_WHIR_ELF"))
        .expect("read the ELF");
    let input = std::fs::read(std::env::var("BLOCK_WHIR_INPUT").expect("BLOCK_WHIR_INPUT"))
        .expect("read the input");
    let cfg = WhirTreeConfig::from_env().unwrap_or_else(|e| panic!("{e}"));
    // The block's base is its proof family's (D-P3B P3). `BLOCK_WHIR_BASE`
    // was this harness's word for it: `p1`/`p1w` are still accepted and
    // change nothing (an A/B against an older binary keeps equal environment
    // bytes); any other value — `rpx` — is refused rather than ignored.
    if let Ok(word) = std::env::var("BLOCK_WHIR_BASE") {
        assert!(
            matches!(word.trim(), "p1" | "p1w"),
            "BLOCK_WHIR_BASE={word}: the block's base is its proof family's (p1)"
        );
    }
    println!("W3 BASE HASH: {:?}", cfg.format.zf.base.hash);
    let run =
        prove_whir_block_tree(&elf, &input, &cfg, &StdoutSink).unwrap_or_else(|e| panic!("{e}"));
    let (proof, plan, proofs, early_out, base) = (
        &run.base_proof,
        &run.plan,
        &run.proofs,
        &run.early,
        run.times.base,
    );
    let top = run.top().expect("a top");
    let (leaves, fan_in, argue, format, early_on) =
        (cfg.leaves, cfg.fan_in, cfg.argue, cfg.format, cfg.early_on);
    let opts = super::proof::block_base_options();
    let wrap = aggregation_wrap_options();
    // Off the clock: the early leaf's arena, built from the groups as phase B
    // handed them over, is the one the finished proof gives.
    match early_out {
        Some((k, arena, complete_at, started, secs)) => {
            let same = block_leaf_arena(plan, proof, *k).expect("the leaf's arena") == *arena;
            println!(
                "W3 EARLY LEAF: leaf {k} complete at {complete_at:.2}s · execute + fill from {started:.2}s for {secs:.2}s (base ended {base:.2}s) · arena = the proof's {same}"
            );
            assert!(same, "the early leaf's arena is the finished proof's");
        }
        None => println!("W3 EARLY LEAF: none (W3_LEAF_DURING_PHASE_B {early_on})"),
    }
    // The recursion proofs' domains and twiddles built before their card holds
    // (`LFM_WARM_DOMAINS`): how many lengths, how long, and when the early
    // leaf's were warm on the tree's clock.
    println!("W3 WARM DOMAINS: {}", super::domain_warm::line());
    // Off the clock: what the device declined, and the card's peaks as the
    // ledger and the memory pool count them (not total − free, which reads a
    // retaining pool's blocks as used).
    #[cfg(feature = "cuda")]
    let (reserved_peak, pool_peak) = (
        format!(
            "{:.2}",
            math_cuda::device::reserved_high_water() as f64 / (1u64 << 30) as f64
        ),
        math_cuda::device::pool_used_bytes().map_or("-".to_string(), |(_, high)| {
            format!("{:.2}", high as f64 / (1u64 << 30) as f64)
        }),
    );
    #[cfg(not(feature = "cuda"))]
    let (reserved_peak, pool_peak) = ("-".to_string(), "-".to_string());
    #[cfg(feature = "cuda")]
    let argue_fallbacks = math_cuda::device::device_fallbacks();
    #[cfg(not(feature = "cuda"))]
    let argue_fallbacks = 0u64;
    println!(
        "W3 DEVICE: commit/encode to the host {} · device commit errors {} · openings on the host {} · argue device fallbacks {argue_fallbacks} · GKR tree refusals {} · fused declines {} · reservation peak {reserved_peak} GiB · pool live peak {pool_peak} GiB",
        multilinear::gpu::host_fallbacks(),
        multilinear::gpu::commit_errors(),
        multilinear::gpu::open_host_fallbacks(),
        multilinear::gpu::gkr_tree_refusals(),
        multilinear::gpu_fused::fused_declines(),
    );
    // Only under the coset round-0 order, so every other run's lines are
    // today's: the round-0 openings served from the stack and the codewords
    // re-encoded after all.
    if crate::zf_format::ZfFormat::global().whir_leaves == multilinear::whir_chain::LeafOrder::Coset
    {
        // Then the openings that read the stack where it lies (the default;
        // `LAMBDA_VM_WHIR_COSET_INPLACE=0` assembles), those asked to whose
        // parts the kernels could not place (assembled after all), and their
        // tiles folded to zero unread and read value by value.
        let (zero, mixed) = multilinear::gpu::coset_inplace_tiles();
        println!(
            "W3 WHIR COSET: round-0 leaves in the coset order · openings from the stack {} · re-encoded {} · in place {} ({}) · assembled after all {} · tiles zero {} · mixed {}",
            multilinear::gpu::coset_openings(),
            multilinear::gpu::coset_fallbacks(),
            multilinear::gpu::coset_inplace_opens(),
            if multilinear::gpu::coset_inplace_in_effect() {
                "on"
            } else {
                "off"
            },
            multilinear::gpu::coset_inplace_assembled(),
            zero,
            mixed,
        );
    }
    // Each table's input layer kept on the card without its padding
    // (`LAMBDA_VM_ARGUE_INPUT_REAL`), and under `check` how many were compared
    // round by round with the whole layer.
    println!(
        "W3 ARGUE INPUT REAL: {} · {}",
        multilinear::gkr_input_real::input_real_in_effect(),
        multilinear::gkr_input_real::counts().line()
    );
    // The argue's round sums through a slot (`LAMBDA_VM_ARGUE_ROUND_SLOT`):
    // rounds read from it, waits that fell back to the stream, today's second
    // launches left, and under `check` the rounds compared with today's.
    println!(
        "W3 ARGUE ROUND SLOT: {} · {}",
        multilinear::round_slot::round_slot_in_effect(),
        multilinear::round_slot::counts_line()
    );
    // The precomputed trees the run downloaded on cache misses, by path, and
    // the staging pairs' waits (I-COPIES L1): the recursion's LFM proofs are
    // where they happen.
    #[cfg(feature = "cuda")]
    println!(
        "W3 TREE DOWNLOAD ({}): {} · staging pair waits {}",
        if math_cuda::device::tree_download_staged() {
            "staged"
        } else {
            "pageable"
        },
        math_cuda::device::tree_download_totals().line(),
        math_cuda::device::staging_totals().pair_waits
    );
    // The base's stacked openings: the lean rounds in effect
    // (`LAMBDA_VM_WHIR_LEAN_ROUNDS`, default 3) and how many openings took them.
    println!(
        "W3 WHIR LEAN: {} rounds · lean opens {} of {}",
        multilinear::gpu::lean_rounds_in_effect(),
        multilinear::gpu::lean_open_calls(),
        multilinear::gpu::open_calls()
    );
    // How those rounds were computed (`LAMBDA_VM_WHIR_LEAN_TILES`): from one
    // pass over the message (the default, `wide`), or over every stack
    // position in every round (`0`); under `check`, the rounds and the bound
    // tables both ways, and how many differed. Then the most one opening held
    // on the card, by form.
    {
        let (compared, differing) = multilinear::gpu::lean_tile_checks();
        let (tables, positions) = multilinear::gpu::lean_tile_table_checks();
        let mib = |bytes: u64| bytes as f64 / (1u64 << 20) as f64;
        let [share, tile, check] = multilinear::gpu::lean_open_peaks();
        println!(
            "W3 WHIR LEAN TILES: {} · tile opens {} of {} lean · rounds compared {compared} · differing {differing} · tables compared {tables} · positions differing {positions} · opening peak MiB share {:.1} tile {:.1} check {:.1}",
            multilinear::gpu::lean_tiles_in_effect(),
            multilinear::gpu::lean_tile_opens(),
            multilinear::gpu::lean_open_calls(),
            mib(share),
            mib(tile),
            mib(check)
        );
    }
    // Each chain's first successor: encoded from the bound message (the
    // default) or folded (`LAMBDA_VM_WHIR_ENCODE_SUCCESSOR=0`), and how many
    // folds ran.
    println!(
        "W3 WHIR SUCCESSORS: encode {} · {} encoded · {} folds ({} fused) · encodes the card declined {}",
        if multilinear::gpu::encode_successor_in_effect() {
            "on"
        } else {
            "off"
        },
        multilinear::gpu::encoded_successors(),
        multilinear::gpu::resident_fold_calls(),
        multilinear::gpu::fused_fold_calls(),
        multilinear::gpu::encode_declines()
    );
    // Coset-ordered round-0 successors encoded on the host because the card
    // declined the opening: 0 on a run whose openings all went to the card.
    println!(
        "W3 WHIR HOST ENCODES: {}",
        multilinear::gpu::coset_host_encodes()
    );
    // How each group's stacked polynomials were opened (`format.group`): one
    // chain a polynomial, or one a run of up to four whose round 0 ran on the
    // card, or fell to the host over the card's codewords (a fallback, each of
    // its polynomials also among the openings on the host above); and the
    // grouped successors folded and combined on the card rather than encoded.
    println!(
        "W3 WHIR GROUPED: {:?} · grouped chains on the card {} · on the host (fallback) {} · successors combined {}",
        format.group,
        multilinear::gpu::grouped_card_chains(),
        multilinear::gpu::grouped_host_chains(),
        multilinear::gpu::grouped_combined_successors()
    );
    // The top proof's bytes: two runs prove the same ones under
    // LAMBDA_VM_FIXED_TRACE_HASH=1 and LAMBDA_VM_DETERMINISTIC_GRIND=1.
    let top_bytes = rkyv::to_bytes::<rkyv::rancor::Error>(&top.proof).expect("serialize the top");
    println!(
        "W3 TOP DIGEST: {} (blake3 of the top proof's {} bytes)",
        &blake3::hash(&top_bytes).to_hex()[..32],
        top_bytes.len()
    );
    // What phase A fixes before any challenge or grind: the group roots, the
    // partition, the tables' heights and counts, the pages and the public
    // output. Two runs over the same tables placed alike give the same one
    // under LAMBDA_VM_FIXED_TRACE_HASH=1 alone, so a block too large to prove
    // twice under the deterministic grind (≈ 23 s of serial grinding a group in
    // phase B) gates its tables' identity on this.
    let mut statement = blake3::Hasher::new();
    for root in &proof.proof.roots {
        statement.update(root);
    }
    for group in &proof.groups {
        statement.update(&(group.len() as u64).to_le_bytes());
        for table in group {
            statement.update(&table.to_le_bytes());
        }
    }
    statement.update(&proof.table_num_vars);
    statement.update(format!("{:?}", proof.table_counts).as_bytes());
    statement.update(format!("{:?}", proof.runtime_page_ranges).as_bytes());
    statement.update(&proof.public_output);
    statement.update(&(proof.num_private_input_pages as u64).to_le_bytes());
    println!(
        "W3 STATEMENT DIGEST: {} (blake3 of {} group roots, {} groups, {} tables' heights and counts)",
        &statement.finalize().to_hex()[..32],
        proof.proof.roots.len(),
        proof.groups.len(),
        proof.table_num_vars.len()
    );

    // The tree's freed pages back to the OS before the verifiers, as
    // `LAMBDA_VM_ALLOC_PURGE` names it (`tree` is not an `auto` point).
    crate::alloc_purge::purge_point("tree");

    // Off the clock: the harness's checks, the permit disarmed first. The
    // verifier derives the tree's programs and artifacts as rayon jobs
    // (`WhirBlockPlan::programs`), which must not take an armed card from a
    // rayon worker (BIG 569); nothing proves beside them now.
    super::device_permit::arm(1);
    let t = std::time::Instant::now();
    for (i, (artifacts, lfm)) in proofs.iter().enumerate() {
        assert!(
            super::proof::verify_against_artifacts_under::<BlockFamily>(
                artifacts,
                &lfm.proof,
                &lfm.public_words,
                &wrap
            ),
            "tree proof {i} verifies against its own program"
        );
    }
    let ok = verify_block_whir(proof, &elf, &opts, &format).expect("the verifier runs");
    println!(
        "W3 HARNESS VERIFY: {} tree proofs and the base {} in {:.2}s",
        proofs.len(),
        if ok { "ACCEPTED" } else { "REJECTED" },
        t.elapsed().as_secs_f64()
    );
    assert!(ok, "the block proof must verify");
    let t = std::time::Instant::now();
    // The production verifier when the run is at its presets, given the run's
    // group opening (`LAMBDA_VM_WHIR_GROUP`) as the CLI's `verify-block --group`
    // gives it; the fixture form only when a knob moved the tree off them.
    let verdict =
        if leaves.is_none() && fan_in == BLOCK_FAN_IN && argue == BlockFormat::production().argue {
            super::whir_block::verify_block_tree_grouped(&elf, proof.statement(), top, format.group)
        } else {
            verify_block_tree_under(
                &elf,
                &opts,
                &format,
                proof.statement(),
                leaves,
                fan_in,
                &wrap,
                top,
            )
        };
    println!(
        "W3 TREE VERIFY: {} in {:.2}s (derives every program and its artifacts)",
        if verdict.is_ok() {
            "ACCEPTED"
        } else {
            "REJECTED"
        },
        t.elapsed().as_secs_f64()
    );
    verdict.expect("the block's verifier accepts the top");

    // `W3_CROSS_SCALE=<T>`, off the clock: the top verified again under the
    // other leaf scale — the single heights when the run's format doubles
    // from some threshold, doubled from `T` when it is fixed. Refused when
    // that scale derives another partition; accepted when it derives the same
    // one (below the threshold).
    if let Ok(word) = std::env::var("W3_CROSS_SCALE") {
        use super::whir_block::LeafScale;
        let t: usize = word
            .trim()
            .parse()
            .unwrap_or_else(|_| panic!("W3_CROSS_SCALE is a leaf count, got `{word}`"));
        assert!(
            leaves.is_none(),
            "W3_CROSS_SCALE: a forced leaf count has no scale"
        );
        let other = BlockFormat {
            leaf_scale: match format.leaf_scale {
                LeafScale::Fixed => LeafScale::DoubleFrom(t),
                LeafScale::DoubleFrom(_) => LeafScale::Fixed,
            },
            ..format
        };
        let clock = std::time::Instant::now();
        let other_plan = WhirBlockPlan::derive(&elf, &opts, &other, proof.statement(), None)
            .expect("the other scale's plan derives");
        let (same, other_leaves) = (
            other_plan.partition() == plan.partition(),
            other_plan.partition().num_leaves(),
        );
        drop(other_plan);
        let crossed = verify_block_tree_under(
            &elf,
            &opts,
            &other,
            proof.statement(),
            None,
            fan_in,
            &wrap,
            top,
        );
        println!(
            "W3 CROSS SCALE: the top under {}: {} · that scale's partition is {} ({other_leaves} leaves against {}) · as expected {} in {:.2}s",
            other.leaf_scale,
            if crossed.is_ok() {
                "ACCEPTED"
            } else {
                "REJECTED"
            },
            if same { "the same" } else { "another" },
            plan.partition().num_leaves(),
            crossed.is_ok() == same,
            clock.elapsed().as_secs_f64()
        );
        assert_eq!(
            crossed.is_ok(),
            same,
            "under {}: accepted exactly when it derives the run's partition",
            other.leaf_scale
        );
    }
}

/// The real block's trace alone, as the streamed prove builds it under the
/// harness's options, each closed group drained instead of committed
/// ([`block_whir::trace_block_host_only`]): the host's own pace, with no card
/// to wait for. The words of [`the_whir_block_tree_on_a_real_block`]; prints
/// the build's stamps and a `BLOCK HOST ONLY` line.
#[test]
#[ignore = "a real block's trace: box only"]
fn the_whir_block_trace_host_only_on_a_real_block() {
    use super::whir_block_tree::WhirTreeConfig;
    let elf = std::fs::read(std::env::var("BLOCK_WHIR_ELF").expect("BLOCK_WHIR_ELF"))
        .expect("read the ELF");
    let input = std::fs::read(std::env::var("BLOCK_WHIR_INPUT").expect("BLOCK_WHIR_INPUT"))
        .expect("read the input");
    let cfg = WhirTreeConfig::from_env().unwrap_or_else(|e| panic!("{e}"));
    let opts = super::proof::block_base_options();
    let (stamps, drained) =
        block_whir::trace_block_host_only(&elf, &input, &opts, &cfg.format, &cfg.options)
            .unwrap_or_else(|e| panic!("{e}"));
    print!("{}", stamps.report());
    println!("BLOCK HOST ONLY: {drained}");
}

// ========================= the batched argue (N-4) ========================

/// [`small_format`] with each group's tables argued together at the format's
/// cap.
fn small_batched() -> BlockFormat {
    BlockFormat {
        argue: ArgueFormat::BATCHED,
        ..small_format()
    }
}

/// ★ Under the batched format every leaf executes over an honest block proof,
/// publishing the plan's id, state and output, and their shares cancel: the
/// batched groups close the block's bus across the leaves. Each group costs the
/// leaf less than its tables argued one by one.
///
/// At blowup 4 ([`at_blowup_4`]): the test format's small stack makes a lone
/// wide table that production cannot form, and at blowup 2 its leaf rows
/// pass a leaf's room.
#[test]
#[ignore = "executes block leaves over small block proofs; box tier"]
fn batched_block_leaves_execute_and_close_the_bus() {
    let format = at_blowup_4(small_batched());
    for (name, leaves) in [
        ("all_instructions_64", Some(3)),
        ("test_commit_4", Some(2)),
        ("test_keccak", None),
        ("test_commit_4", Some(1)),
    ] {
        let (elf, proof) = small_block(name, &format);
        assert_eq!(proof.argues.len(), proof.groups.len());
        let plan = plan_of(&elf, &proof, &format, leaves);
        let per_table = plan_of(&elf, &proof, &at_blowup_4(small_format()), leaves);
        println!(
            "WHIR BLOCK BATCHED LEAVES {name}: {} groups, costs {:?} (per table {:?}), partition {:?}",
            plan.num_groups(),
            plan.costs(),
            per_table.costs(),
            plan.partition().leaves(),
        );
        for (g, (batched, each)) in plan.costs().iter().zip(per_table.costs()).enumerate() {
            assert!(batched <= each, "{name} group {g}: {batched} > {each}");
        }
        let layout = plan.child_layout();
        let words = leaves_words(&plan, &proof);
        let id = id_words(plan.id());
        let mut sum = FEE::zero();
        for w in &words {
            assert_eq!(w[layout.id(0)], id[0]);
            assert_eq!(w[layout.id(1)], id[1]);
            assert_eq!(w[layout.state(0)], words[0][layout.state(0)]);
            sum += ext_of(&w[layout.sum()]);
        }
        assert_eq!(sum, FEE::zero(), "{name}: the leaves' shares cancel");
        // The format is the plan's: a per-table plan has no reading of a
        // batched proof, and a batched plan none of a per-table one.
        assert!(block_leaf_arena(&per_table, &proof, 0).is_err());
    }
}

/// ★ A batched leaf refuses each forgery of its group's argue — a bus output,
/// a ladder step's half, a ladder round, a constraint round, a factor value —
/// and a tampered prepared opening, which executes with the prepared openings
/// left out (the mutation): the preprocessed and prepared checks stay in place
/// under the batched argue.
#[test]
#[ignore = "executes block leaves over small block proofs; box tier"]
fn a_batched_block_leaf_refuses_a_tampered_witness() {
    let format = small_batched();
    let (elf, proof) = small_block("test_commit_4", &format);
    let plan = plan_of(&elf, &proof, &format, Some(1));
    run_leaf(&plan, &proof, 0, LeafChecks::ALL).expect("the honest leaf executes");
    let one = FEE::one();
    // The tallest bin of group 0, whose ladder has a step with rounds.
    let (bin, steps) = proof.argues[0]
        .gkr
        .iter()
        .enumerate()
        .map(|(i, l)| (i, l.layers.len()))
        .max_by_key(|&(_, steps)| steps)
        .expect("a ladder");
    assert!(steps >= 2, "a ladder with a step that has rounds");
    type Forge = fn(&mut BlockWhirProof, usize, usize);
    let forgeries: [(&str, Forge); 5] = [
        ("bus output", |p, _, _| {
            p.argues[0].bus_outputs[0].0 += FEE::one()
        }),
        ("ladder half", |p, bin, steps| {
            p.argues[0].gkr[bin].layers[steps - 1].halves[0].p_lo += FEE::one()
        }),
        ("ladder round", |p, bin, steps| {
            p.argues[0].gkr[bin].layers[steps - 1].sumcheck.rounds[0].evaluations[0] += FEE::one()
        }),
        ("constraint round", |p, _, _| {
            p.argues[0].constraint.rounds[0].evaluations[0] += FEE::one()
        }),
        ("factor value", |p, _, _| {
            p.argues[0].factor_values[0][0] += FEE::one()
        }),
    ];
    for (name, forge) in forgeries {
        let mut forged = proof.clone();
        forge(&mut forged, bin, steps);
        assert!(
            run_leaf(&plan, &forged, 0, LeafChecks::ALL).is_err(),
            "a forged {name}"
        );
    }
    let mut prepared = proof.clone();
    prepared.prepared[0].polys[0].final_value += one;
    assert!(run_leaf(&plan, &prepared, 0, LeafChecks::ALL).is_err());
    let without = LeafChecks {
        prepared: false,
        ..LeafChecks::ALL
    };
    assert!(
        run_leaf(&plan, &prepared, 0, without).is_ok(),
        "with the prepared openings left out the tamper executes"
    );
}

/// ★ The batched block's tree proves to the top its plan derives, and the
/// format is the verifier's: the batched top is accepted under the batched
/// format and refused under the per-table one.
#[test]
#[ignore = "proves a block tree; box tier"]
fn the_batched_whir_block_tree_proves_to_the_derived_top() {
    super::device_permit::arm(1);
    let format = small_batched();
    let (elf, proof) = small_block("test_commit_4", &format);
    let opts = ProofOptions::default_test_options();
    let wrap = aggregation_wrap_options();
    let plan = plan_of(&elf, &proof, &format, Some(2));
    let (top, _) = compose(&plan, &proof, "BATCHED TREE").expect("the honest tree proves");
    let under = |format: &BlockFormat| {
        verify_block_tree_under(
            &elf,
            &opts,
            format,
            proof.statement(),
            Some(2),
            BLOCK_FAN_IN,
            &wrap,
            &top,
        )
    };
    under(&format).expect("the batched verifier accepts the honest top");
    assert!(
        under(&small_format()).is_err(),
        "the per-table verifier refuses the batched top"
    );
}

/// ★ The grind bits (`LAMBDA_VM_ZF_WHIR_GRIND_BITS`) are a verifier constant
/// on the block's recursion too: the statement, Q and the grind are the leaf
/// program's constants. Over a block ground at 18 bits the plan at 18 emits a
/// leaf that executes; the plan at 20 emits one that refuses the same proof,
/// and executes over a block ground at 20 (the control: the refusal is the
/// bits', not the leaf's). Every leaf runs on the family's socket, so no
/// refusal is the hasher's (REV-P3B14-A A4). Q moves with the bits (114 at 18,
/// 112 at 20), so the refusal may be the arena's layout before any grind is
/// checked; the line `GRIND LEAF REFUSAL:` says which.
#[test]
#[ignore = "executes block leaves over small block proofs; box tier"]
fn a_block_leaf_at_20_bits_refuses_a_block_ground_at_18() {
    let at = |bits: u8| BlockFormat {
        zf: ZfFormat {
            whir_grind_bits: bits,
            ..small_format().zf
        },
        ..small_format()
    };
    let (elf, proof) = small_block("test_commit_4", &at(18));
    let plan = plan_of(&elf, &proof, &at(18), Some(1));
    run_leaf(&plan, &proof, 0, LeafChecks::ALL).expect("the leaf at 18 bits executes");

    let strict = plan_of(&elf, &proof, &at(20), Some(1));
    assert_ne!(
        strict.id(),
        plan.id(),
        "the statement at 20 bits is another statement"
    );
    let program = leaf_program_with(&strict, 0, LeafChecks::ALL).expect("the leaf at 20 emits");
    let socket = program.hasher(<BlockFamily as ProofFamily>::SOCKET);
    assert_eq!(
        socket,
        super::hash::HasherKind::Poseidon1W16,
        "the leaf's socket"
    );
    let refusal = match block_leaf_arena(&strict, &proof, 0) {
        Err(e) => format!("the arena: {e}"),
        Ok(arenas) if program.arena_schema.lens != vec![arenas[0].len() as u32] => format!(
            "the arena's length: {} words for a leaf hinting {:?}",
            arenas[0].len(),
            program.arena_schema.lens
        ),
        Ok(arenas) => match execute(&program, &arenas, &socket) {
            Err(e) => {
                assert!(
                    !matches!(
                        e,
                        super::executor::LfmExecError::HasherRejected(_)
                            | super::executor::LfmExecError::Internal(_)
                    ),
                    "the leaf at 20 refused for another reason than the proof: {e:?}"
                );
                format!("the execution: {e:?}")
            }
            Ok(_) => panic!("a leaf at 20 bits executed over a block ground at 18"),
        },
    };
    println!("GRIND LEAF REFUSAL: {refusal}");
    // The control: the same leaf at 20 executes over a block ground at 20.
    let (elf20, proof20) = small_block("test_commit_4", &at(20));
    let plan20 = plan_of(&elf20, &proof20, &at(20), Some(1));
    run_leaf(&plan20, &proof20, 0, LeafChecks::ALL)
        .expect("the leaf at 20 executes over a block at 20");
}

/// ★ The base code's rate (`LAMBDA_VM_ZF_WHIR_BLOWUP`) is a verifier constant
/// on the block's recursion too: the statement, the domains, the tree depths
/// and Q are the leaf program's constants. Each way round, the plan at the
/// block's own blowup emits a leaf that executes over it, and the plan at the
/// other blowup is another statement whose leaf refuses the same proof — at its
/// arena (other path lengths and Q) or in its execution, never at the hasher
/// (every leaf runs on the family's socket). The line `RATE LEAF REFUSAL:`
/// says where.
#[test]
#[ignore = "executes block leaves over small block proofs; box tier"]
fn a_block_leaf_at_one_blowup_refuses_a_block_at_the_other() {
    let at = |blowup: u8| BlockFormat {
        zf: ZfFormat {
            whir_blowup: blowup,
            ..small_format().zf
        },
        ..small_format()
    };
    for (own, other) in [(2u8, 4u8), (4, 2)] {
        let (elf, proof) = small_block("test_commit_4", &at(own));
        let plan = plan_of(&elf, &proof, &at(own), Some(1));
        run_leaf(&plan, &proof, 0, LeafChecks::ALL)
            .unwrap_or_else(|e| panic!("the leaf at blowup {own} executes over its block: {e}"));

        let wrong = plan_of(&elf, &proof, &at(other), Some(1));
        assert_ne!(
            wrong.id(),
            plan.id(),
            "the statement at blowup {other} is another statement"
        );
        let program =
            leaf_program_with(&wrong, 0, LeafChecks::ALL).expect("the leaf at the other blowup");
        let socket = program.hasher(<BlockFamily as ProofFamily>::SOCKET);
        let refusal = match block_leaf_arena(&wrong, &proof, 0) {
            Err(e) => format!("the arena: {e}"),
            Ok(arenas) if program.arena_schema.lens != vec![arenas[0].len() as u32] => format!(
                "the arena's length: {} words for a leaf hinting {:?}",
                arenas[0].len(),
                program.arena_schema.lens
            ),
            Ok(arenas) => match execute(&program, &arenas, &socket) {
                Err(e) => {
                    assert!(
                        !matches!(
                            e,
                            super::executor::LfmExecError::HasherRejected(_)
                                | super::executor::LfmExecError::Internal(_)
                        ),
                        "the leaf at blowup {other} refused for another reason than the \
                         proof: {e:?}"
                    );
                    format!("the execution: {e:?}")
                }
                Ok(_) => panic!("a leaf at blowup {other} executed over a block at blowup {own}"),
            },
        };
        println!("RATE LEAF REFUSAL: a block at blowup {own}, a leaf at {other}: {refusal}");
    }
}

/// I-GRIND2 instrument (not a gate): a block group's in-guest opening cost
/// ([`super::whir_stacked::stacked_verify_cost`], what the plan charges a
/// group for its stacked opening and each prepared one) at 20 grind bits
/// (Q 112) and at `LAMBDA_VM_ZF_WHIR_GRIND_BITS=18` (Q 114), per stack height
/// and polynomial count. The per-group deltas predict the W3 plan's costs and
/// leaf count under the arm; the argue's cost does not read Q.
///
/// cargo test -p lambda-vm-prover --lib grind_bits_group_cost_census -- --ignored --nocapture
#[test]
#[ignore = "instrument: prints the group opening cost per setting, asserts nothing"]
fn grind_bits_group_cost_census() {
    for bits in crate::zf_format::WHIR_GRIND_BITS {
        let format = ZfFormat {
            whir_grind_bits: bits,
            ..ZfFormat::DEFAULT
        };
        for (polys, n) in [1usize, 2, 3]
            .into_iter()
            .map(|p| (p, 27usize))
            .chain((15..=26).map(|n| (1, n)))
        {
            // Tables of 2^n cells, one per polynomial: at 27 the stack cap
            // keeps them apart, below it one table is one polynomial.
            let height = n.min(21);
            let shapes = vec![(1usize << (n - height), height); polys];
            let config = crate::multilinear_prove::chain_config_under(&format, &shapes);
            let layout = stark::multilinear_table::global_layout(&shapes, config.format.stack)
                .expect("a layout");
            let (_, group_of) = super::whir_epoch::group_columns(&shapes);
            let chain = super::whir_chain::ChainShape::new(&config, layout.n_stack());
            let cost = super::whir_stacked::stacked_verify_cost(
                &layout,
                &group_of,
                &chain,
                super::whir_epoch::fresh_schedule().entry(),
            );
            println!(
                "GRINDGROUP bits={bits} Q={} n_stack={} polys={} perms={} instrs={}",
                config.num_queries,
                layout.n_stack(),
                layout.num_polys(),
                cost.perms(),
                cost.operations(),
            );
        }
    }
}

// ======================== the Poseidon1 base (S5) =========================

/// ★ The Poseidon1 leaves' front draws the host verifier's challenges:
/// `P1Transcript` over the statement (call by call), every root, every
/// commit-time answer, `z, α, β`, a group fork's first draw, and that group's
/// commit-time point (`multilinear_block::commit_ood_point`). The host side is
/// spelled out here, so a misplaced answer on either side is a mismatch: each
/// of the front's mutations — the answers after the draws, the first group's
/// answers left out — moves the draws off the host's.
#[test]
#[ignore = "executes a block leaf's front over a small block proof; box tier"]
fn the_p1_block_front_draws_the_hosts_challenges() {
    use super::whir_block::ood_faults::{self, Fault};
    use crypto::fiat_shamir::is_transcript::IsTranscript;
    use crypto::fiat_shamir::p1_transcript::P1Transcript;
    let format = small_format();
    let (elf, proof) = small_block("test_commit_4", &format);
    let plan = plan_of(&elf, &proof, &format, Some(1));
    assert_eq!(plan.wrap_hash(), super::edsl::WrapHash::Poseidon1);
    let g = plan.num_groups() - 1;
    let mut arena: Vec<LfmWord> = proof
        .proof
        .roots
        .iter()
        .map(super::algebraic_commit::commitment_to_digest)
        .collect();
    assert_eq!(proof.commit_ood.len(), proof.proof.roots.len());
    arena.extend(proof.commit_ood.iter().map(super::word::ext_word));
    let draws = |fault: Fault| -> Vec<FEE> {
        let program = ood_faults::with(fault, || super::whir_block::front_program(&plan, g));
        assert!(program.hash16(), "the front's hash rows are the socket's");
        execute(
            &program,
            std::slice::from_ref(&arena),
            &program.hasher(<BlockFamily as ProofFamily>::SOCKET),
        )
        .expect("the front executes")
        .public_words
        .iter()
        .map(|(_, w)| ext_of(w))
        .collect()
    };
    let machine = draws(Fault::None);
    let opts = ProofOptions::default_test_options();
    let frame = block_whir::block_frame(proof.statement(), &elf, &opts, &format).expect("frame");
    let derived: Vec<multilinear::whir_commit::Commitment> = plan
        .prepared()
        .iter()
        .flat_map(|p| p.roots.iter().copied())
        .collect();
    let mut t = P1Transcript::new();
    block_whir::absorb_block(
        &mut t,
        &block_whir::block_statement_tag(&format.zf.base),
        &elf,
        &proof.public_output,
        &proof.table_counts,
        proof.num_private_input_pages,
        &proof.runtime_page_ranges,
        &proof.table_num_vars,
        &frame.config,
        &proof.groups,
    );
    for root in proof.proof.roots.iter().chain(&derived) {
        t.append_bytes(root);
    }
    for answer in &proof.commit_ood {
        t.append_field_element(answer);
    }
    let mut host: Vec<FEE> = (0..3).map(|_| t.sample_field_element()).collect();
    let mut fork = t.clone();
    fork.append_bytes(&(g as u64).to_le_bytes());
    host.push(fork.sample_field_element());
    let layout = &frame.stack_layouts[g];
    let root_at: usize = frame.stack_layouts[..g]
        .iter()
        .map(multilinear::stacking::StackedLayout::num_polys)
        .sum();
    host.push(
        stark::multilinear_block::commit_ood_point::<FEE_FIELD, P1Transcript>(
            g,
            layout,
            &proof.proof.roots[root_at..root_at + layout.num_polys()],
            &frame.config,
        )
        .expect("the group's point"),
    );
    assert_eq!(
        machine, host,
        "z, α, β, the fork's first draw and the group's point"
    );
    for fault in [Fault::AnswersAfterDraws, Fault::SkipFirstGroupsAnswers] {
        assert_ne!(
            draws(fault)[..3],
            host[..3],
            "{fault:?}: the front's draws must move off the host's"
        );
    }
}

/// ★ A Poseidon1 leaf refuses a tampered witness: a group root, a table's
/// argument, a Merkle sibling and a grind nonce of a group opening, and a
/// prepared opening (the prepared check is the one refusing it).
#[test]
#[ignore = "executes block leaves over small block proofs; box tier"]
fn a_p1_block_leaf_refuses_a_tampered_witness() {
    use multilinear::whir_chain::RoundOpenings;
    let format = small_format();
    let (elf, proof) = small_block("test_commit_4", &format);
    let plan = plan_of(&elf, &proof, &format, Some(1));
    run_leaf(&plan, &proof, 0, LeafChecks::ALL).expect("the honest leaf executes");

    let mut root = proof.clone();
    root.proof.roots[0][0] ^= 1;
    assert!(
        run_leaf(&plan, &root, 0, LeafChecks::ALL).is_err(),
        "a group root"
    );

    let mut table = proof.clone();
    table.proof.tables[0].bus_output.0 += FEE::one();
    assert!(
        run_leaf(&plan, &table, 0, LeafChecks::ALL).is_err(),
        "a table argument"
    );

    let mut sibling = proof.clone();
    match &mut sibling.proof.columns[0].polys[0].rounds[0].openings {
        RoundOpenings::Base(p) => p.current[1].proof.merkle_path[0][0] ^= 1,
        RoundOpenings::Extension(p) => p.current[1].proof.merkle_path[0][0] ^= 1,
        RoundOpenings::BaseGrouped(_) => unreachable!("a per-polynomial chain"),
    }
    assert!(
        run_leaf(&plan, &sibling, 0, LeafChecks::ALL).is_err(),
        "a Merkle sibling"
    );

    let mut nonce = proof.clone();
    nonce.proof.columns[0].polys[0].rounds[0].nonces.query ^= 1;
    assert!(
        run_leaf(&plan, &nonce, 0, LeafChecks::ALL).is_err(),
        "a grind nonce"
    );

    // A commit-time answer, first and last: every leaf absorbs every answer
    // in its front, and a group's chains carry its own.
    for at in [0, proof.commit_ood.len() - 1] {
        let mut answer = proof.clone();
        answer.commit_ood[at] += FEE::one();
        assert!(
            run_leaf(&plan, &answer, 0, LeafChecks::ALL).is_err(),
            "commit-time answer {at}"
        );
    }

    let mut prepared = proof.clone();
    prepared.prepared[0].polys[0].final_value += FEE::one();
    assert!(
        run_leaf(&plan, &prepared, 0, LeafChecks::ALL).is_err(),
        "a prepared opening"
    );
    let without = LeafChecks {
        prepared: false,
        ..LeafChecks::ALL
    };
    assert!(
        run_leaf(&plan, &prepared, 0, without).is_ok(),
        "with the prepared openings left out the tamper executes"
    );
}

/// ★ Mutation: a group's walk that reads
/// its commit-time answers from a fresh hint rather than the cells the front
/// absorbed hints words the block's arena does not hold: the leaf's arena is
/// the proof's, one answer a root, read once, by the front.
#[test]
#[ignore = "executes block leaves over small block proofs; box tier"]
fn a_walk_reads_its_answers_from_the_fronts_cells() {
    use super::whir_block::ood_faults::{self, Fault};
    let format = small_format();
    let (elf, proof) = small_block("test_commit_4", &format);
    let plan = plan_of(&elf, &proof, &format, Some(1));
    let honest = leaf_program_with(&plan, 0, LeafChecks::ALL).expect("the leaf");
    let arenas = block_leaf_arena(&plan, &proof, 0).expect("the arena");
    assert_eq!(honest.arena_schema.lens, vec![arenas[0].len() as u32]);
    let fresh = ood_faults::with(Fault::FreshAnswerHint, || {
        leaf_program_with(&plan, 0, LeafChecks::ALL).expect("the mutated leaf")
    });
    assert_ne!(
        fresh.arena_schema.lens,
        vec![arenas[0].len() as u32],
        "a walk reading its answers anew hints words the proof's arena does not hold"
    );
}

// ============ D-ONECHAIN B2: the leaves over an `Indexed` block ============

/// [`small_format`] at three polynomials a group under `base`, each group's
/// tables argued per `argue` and its stacked polynomials opened by one chain
/// per run of them (`GroupFold::Indexed`).
fn indexed_format(base: stark::proof::options::BaseFormat, argue: ArgueFormat) -> BlockFormat {
    let small = small_format();
    BlockFormat {
        zf: small.zf.with_base(base),
        group_polys: 3,
        argue,
        group: multilinear::whir_chain::GroupFold::Indexed,
        ..small
    }
}

/// ★ REV-B2-JUDGE B2-6 at the block (D-ONECHAIN B2), under the block's base:
/// the plan's split is the proof's (one chain per run), every leaf of an
/// `Indexed` block executes over the honest proof and the leaves' shares
/// cancel; a grouped round-0 value tampered in a group is refused by the host
/// and by that group's leaf; and the plan prices each group at most at its
/// per-polynomial cost, below it wherever a run opens several trees.
#[test]
#[ignore = "executes block leaves over small block proofs; box tier"]
fn an_indexed_blocks_leaves_execute_and_refuse_a_grouped_forgery() {
    indexed_leaves_execute_and_refuse_a_grouped_forgery(ArgueFormat::PerTable);
}

/// [`an_indexed_blocks_leaves_execute_and_refuse_a_grouped_forgery`] with each
/// group's tables argued together (`ArgueFormat::BATCHED`), the argue
/// production runs: the pairing a default would ship, every leaf executed.
#[test]
#[ignore = "executes block leaves over small block proofs; box tier"]
fn an_indexed_blocks_leaves_execute_under_the_batched_argue() {
    indexed_leaves_execute_and_refuse_a_grouped_forgery(ArgueFormat::BATCHED);
}

/// ★ The base code's rate is the leaf's constant under one chain a run too, at
/// the batched argue (the composed format): each way round, every leaf of the
/// plan at the block's own blowup executes, and the plan at the other blowup —
/// another statement — has a first leaf that refuses the block at its arena or
/// in its execution, never at the hasher. The line `RATE INDEXED LEAF REFUSAL:`
/// says where.
#[test]
#[ignore = "executes block leaves over small block proofs; box tier"]
fn an_indexed_blocks_leaves_at_one_blowup_refuse_a_block_at_the_other() {
    let at = |blowup: u8| {
        let mut f = indexed_format(crate::hash_pin::BLOCK_BASE, ArgueFormat::BATCHED);
        f.zf.whir_blowup = blowup;
        f
    };
    for (own, other) in [(2u8, 4u8), (4, 2)] {
        let (elf, proof) = small_block("all_instructions_64", &at(own));
        let plan = plan_of(&elf, &proof, &at(own), Some(2));
        for k in 0..plan.partition().num_leaves() {
            run_leaf(&plan, &proof, k, LeafChecks::ALL).unwrap_or_else(|e| {
                panic!("leaf {k} at blowup {own} executes over its Indexed block: {e}")
            });
        }
        let wrong = plan_of(&elf, &proof, &at(other), Some(2));
        assert_ne!(wrong.id(), plan.id(), "another statement at blowup {other}");
        let program =
            leaf_program_with(&wrong, 0, LeafChecks::ALL).expect("the leaf at the other blowup");
        let socket = program.hasher(<BlockFamily as ProofFamily>::SOCKET);
        let refusal = match block_leaf_arena(&wrong, &proof, 0) {
            Err(e) => format!("the arena: {e}"),
            Ok(arenas) if program.arena_schema.lens != vec![arenas[0].len() as u32] => format!(
                "the arena's length: {} words for a leaf hinting {:?}",
                arenas[0].len(),
                program.arena_schema.lens
            ),
            Ok(arenas) => match execute(&program, &arenas, &socket) {
                Err(e) => {
                    assert!(
                        !matches!(
                            e,
                            super::executor::LfmExecError::HasherRejected(_)
                                | super::executor::LfmExecError::Internal(_)
                        ),
                        "the leaf at blowup {other} refused for another reason than the \
                         proof: {e:?}"
                    );
                    format!("the execution: {e:?}")
                }
                Ok(_) => panic!(
                    "an Indexed leaf at blowup {other} executed over a block at blowup {own}"
                ),
            },
        };
        println!(
            "RATE INDEXED LEAF REFUSAL: a block at blowup {own}, a leaf at {other}: {refusal}"
        );
    }
}

/// ★ The group opening is the leaf's constant too: a plan derived for the
/// other mode, over a block proved under one, has no leaf that executes the
/// proof's groups — the leaf programs of one mode refuse the other mode's
/// base, both ways, whichever check or arena shape refuses first. The honest
/// pairs execute.
#[test]
#[ignore = "proves small blocks and executes their leaves; box tier"]
fn a_leaf_of_one_group_opening_refuses_the_others_base() {
    use multilinear::whir_chain::GroupFold;
    let base = crate::hash_pin::BLOCK_BASE;
    let indexed = indexed_format(base, ArgueFormat::BATCHED);
    let per_poly = BlockFormat {
        group: GroupFold::PerPoly,
        ..indexed
    };
    for (label, proved, other) in [
        ("an Indexed base under PerPoly leaves", &indexed, &per_poly),
        ("a PerPoly base under Indexed leaves", &per_poly, &indexed),
    ] {
        let (elf, proof) = small_block("all_instructions_64", proved);
        let honest = plan_of(&elf, &proof, proved, Some(2));
        for k in 0..honest.partition().num_leaves() {
            run_leaf(&honest, &proof, k, LeafChecks::ALL)
                .unwrap_or_else(|e| panic!("{label}: the honest leaf {k} refused: {e}"));
        }
        let crossed = plan_of(&elf, &proof, other, Some(2));
        // `run_leaf` without its assertion: an arena of another length than
        // the program hints is one more way the crossed leaf refuses.
        let crossed_leaf = |k: usize| -> Result<(), String> {
            let program = leaf_program_with(&crossed, k, LeafChecks::ALL)?;
            let arenas = block_leaf_arena(&crossed, &proof, k)?;
            if program.arena_schema.lens != vec![arenas[0].len() as u32] {
                return Err(format!(
                    "the arena holds {} words, the program hints {:?}",
                    arenas[0].len(),
                    program.arena_schema.lens
                ));
            }
            execute(
                &program,
                &arenas,
                &program.hasher(<BlockFamily as ProofFamily>::SOCKET),
            )
            .map(|_| ())
            .map_err(|e| format!("{e:?}"))
        };
        // A hint builder that cannot read the other mode's openings may stop
        // by a panic rather than an error: the leaf has no arena either way.
        let crossed_leaf = |k: usize| -> Result<(), String> {
            std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| crossed_leaf(k)))
                .unwrap_or_else(|_| Err("the hint builder stopped (a panic)".to_string()))
        };
        let refused = (0..crossed.partition().num_leaves())
            .filter(|&k| crossed_leaf(k).is_err())
            .count();
        println!(
            "ONECHAIN CROSS LEAF {label}: {refused} of {} leaves refuse",
            crossed.partition().num_leaves()
        );
        assert!(
            refused > 0,
            "{label}: a leaf of the other mode executed the base"
        );
        // Every leaf over a group the two modes open differently — one of
        // two or more polynomials — refuses.
        let frame = block_whir::block_frame(
            proof.statement(),
            &elf,
            &ProofOptions::default_test_options(),
            proved,
        )
        .expect("the frame");
        let differ: Vec<usize> = frame
            .stack_layouts
            .iter()
            .enumerate()
            .filter(|(_, layout)| layout.num_polys() >= 2)
            .map(|(g, _)| g)
            .collect();
        assert!(
            !differ.is_empty(),
            "{label}: a group of several polynomials"
        );
        for k in 0..crossed.partition().num_leaves() {
            if crossed
                .partition()
                .leaf(k)
                .iter()
                .any(|g| differ.contains(g))
            {
                assert!(
                    crossed_leaf(k).is_err(),
                    "{label}: leaf {k} holds a group the modes open differently and executed"
                );
            }
        }
    }
}

fn indexed_leaves_execute_and_refuse_a_grouped_forgery(argue: ArgueFormat) {
    use multilinear::whir_chain::RoundOpenings;
    let base = crate::hash_pin::BLOCK_BASE;
    let format = indexed_format(base, argue);
    let (elf, proof) = small_block("all_instructions_64", &format);
    let plan = plan_of(&elf, &proof, &format, Some(2));
    let grouped: Vec<(usize, usize)> = proof
        .proof
        .columns
        .iter()
        .enumerate()
        .flat_map(|(g, stack)| {
            stack
                .polys
                .iter()
                .enumerate()
                .filter(|(_, chain)| {
                    matches!(chain.rounds[0].openings, RoundOpenings::BaseGrouped(_))
                })
                .map(move |(i, _)| (g, i))
        })
        .collect();
    assert!(!grouped.is_empty(), "{base:?}: a grouped chain");

    let words = leaves_words(&plan, &proof);
    let layout = plan.child_layout();
    let sum = words
        .iter()
        .fold(FEE::zero(), |acc, w| acc + ext_of(&w[layout.sum()]));
    assert_eq!(sum, FEE::zero(), "{base:?}: the leaves' shares cancel");

    let per_poly = BlockFormat {
        group: multilinear::whir_chain::GroupFold::PerPoly,
        ..format
    };
    let apart = plan_of(&elf, &proof, &per_poly, Some(2));
    for (g, (&one, &many)) in plan.costs().iter().zip(apart.costs()).enumerate() {
        let runs = proof.proof.columns[g].polys.len();
        assert!(one <= many, "{base:?} group {g}: {one} against {many}");
        if grouped.iter().any(|&(gg, _)| gg == g) {
            assert!(one < many, "{base:?} group {g}: {runs} chains");
        }
    }
    println!(
        "ONECHAIN BLOCK {base:?}: {} groups, grouped chains {grouped:?}, costs {:?} (per \
             polynomial {:?}), leaves {:?}",
        plan.num_groups(),
        plan.costs(),
        apart.costs(),
        plan.partition().leaves()
    );

    // A value of tree 1's block at query 1 in the first grouped chain: the
    // host refuses it at that chain's opening (reported as the run's first
    // polynomial), and the leaf holding the group at tree 1's opening at
    // query 1 — the first failing check in program order, against the
    // honest leaf, which executes.
    let (g, i) = grouped[0];
    let mut forged = proof.clone();
    match &mut forged.proof.columns[g].polys[i].rounds[0].openings {
        RoundOpenings::BaseGrouped(round) => round.current[1][1].values[0] += FE::one(),
        _ => unreachable!("a grouped round 0"),
    }
    let opts = ProofOptions::default_test_options();
    let frame = block_whir::block_frame(proof.statement(), &elf, &opts, &format).expect("frame");
    let run = multilinear::stacked_eval::index_chains(
        frame.stack_layouts[g].num_polys(),
        multilinear::whir_chain::GroupFold::Indexed,
    )[i]
        .clone();
    let refusal = block_whir::verify_block_whir_traced(
        &forged,
        &elf,
        &opts,
        &format,
        false,
        stark::multilinear_table::batched::VerifierChecks::ALL,
    )
    .expect("the statement frames");
    assert!(
        matches!(
            refusal,
            Err(block_whir::BlockRefusal::Proof(
                multilinear::Error::ColumnOpeningRejected { column }
            )) if column == run.start
        ),
        "{base:?}: the host refuses group {g} chain {i}'s opening, got {refusal:?}"
    );
    let k = (0..plan.partition().num_leaves())
        .find(|&k| plan.partition().leaf(k).contains(&g))
        .expect("a leaf holds the group");
    use super::whir_chain::onechain_steps::{self, Step};
    let (program, marks) = onechain_steps::record(|| {
        leaf_program_with(&plan, k, LeafChecks::ALL).expect("the leaf emits")
    });
    let hasher = program.hasher(crate::hash_pin::BLOCK_HASHER);
    let honest = block_leaf_arena(&plan, &proof, k).expect("the honest arena");
    super::executor::execute_serial(&program, &honest, &hasher).expect("the honest leaf executes");
    let forged_arena = block_leaf_arena(&plan, &forged, k).expect("the forged arena");
    let error = super::executor::execute_serial(&program, &forged_arena, &hasher)
        .expect_err("the leaf refuses the forgery");
    assert_eq!(
        onechain_steps::failed_at(&program, &marks, &error),
        Some((
            Some(run.start),
            Step::Opening {
                round: 0,
                query: 1,
                tree: 1
            }
        )),
        "{base:?}: leaf {k} refuses group {g} chain {i} at tree 1's opening, got {error:?}"
    );
    assert!(
        run_leaf(&plan, &forged, k, LeafChecks::ALL).is_err(),
        "{base:?}: and on the production schedule"
    );
}

/// The code review's M-g: a small `Indexed` block under a stack of
/// `2^stack` and `group_polys` polynomials a group, its prepared stack forged
/// in one chain, at the block and in its leaf.
///
/// The block proves and verifies ([`small_block`]); every leaf executes and the
/// shares cancel. `pick` chooses, from the prepared stacks' chains — `(stack,
/// chain, width)`, a width of 1 being a per-polynomial chain — the one to
/// forge: a value of its round-0 block at query 1, in tree 1 of a grouped
/// chain or the one tree of another. The host must refuse at that stack's
/// opening, reported as the run's first polynomial, and the leaf holding the
/// stack's group at that tree's opening at query 1 under the chain's own mark.
/// The forgery touches the prepared opening alone, and the honest leaf
/// executes.
fn forged_prepared_chain(
    base: stark::proof::options::BaseFormat,
    argue: ArgueFormat,
    stack: usize,
    group_polys: usize,
    pick: impl Fn(&[(usize, usize, usize)]) -> Option<(usize, usize, usize)>,
) {
    use super::whir_chain::onechain_steps::{self, Step};
    use multilinear::whir_chain::{GroupFold, RoundOpenings};
    let small = indexed_format(base, argue);
    let mut zf = small.zf;
    zf.whir_stack = StackVars::new(stack).expect("a valid stack");
    let format = BlockFormat {
        zf,
        group_polys,
        ..small
    };
    let (elf, proof) = small_block("all_instructions_64", &format);
    let chains: Vec<(usize, usize, usize)> = proof
        .prepared
        .iter()
        .enumerate()
        .flat_map(|(k, opening)| {
            opening.polys.iter().enumerate().map(move |(i, chain)| {
                let width = match &chain.rounds[0].openings {
                    RoundOpenings::BaseGrouped(g) => g.current.first().map_or(0, Vec::len),
                    _ => 1,
                };
                (k, i, width)
            })
        })
        .collect();
    println!(
        "ONECHAIN PREPARED {:?} stack 2^{stack}: {} groups, prepared chains (stack, chain, width) \
         {chains:?}",
        base.hash,
        proof.groups.len()
    );
    let (k, i, width) = pick(&chains).unwrap_or_else(|| {
        panic!(
            "{:?} stack 2^{stack}: the prepared stacks' chains {chains:?} hold none to forge",
            base.hash
        )
    });

    let plan = plan_of(&elf, &proof, &format, None);
    let words = leaves_words(&plan, &proof);
    let layout = plan.child_layout();
    let sum = words
        .iter()
        .fold(FEE::zero(), |acc, w| acc + ext_of(&w[layout.sum()]));
    assert_eq!(
        sum,
        FEE::zero(),
        "{:?}: the leaves' shares cancel",
        base.hash
    );

    let prepared = &plan.prepared()[k];
    let run =
        multilinear::stacked_eval::index_chains(prepared.layout.num_polys(), GroupFold::Indexed)[i]
            .clone();
    assert_eq!(run.len(), width, "the proof's chain is the derived run");
    let mut forged = proof.clone();
    let tree = match &mut forged.prepared[k].polys[i].rounds[0].openings {
        RoundOpenings::BaseGrouped(round) => {
            round.current[1][1].values[0] += FE::one();
            1
        }
        RoundOpenings::Base(round) => {
            round.current[1].values[0] += FE::one();
            0
        }
        RoundOpenings::Extension(_) => unreachable!("round 0 is base-field"),
    };
    let opts = ProofOptions::default_test_options();
    let refusal = block_whir::verify_block_whir_traced(
        &forged,
        &elf,
        &opts,
        &format,
        false,
        stark::multilinear_table::batched::VerifierChecks::ALL,
    )
    .expect("the statement frames");
    assert!(
        matches!(
            refusal,
            Err(block_whir::BlockRefusal::Proof(
                multilinear::Error::ColumnOpeningRejected { column }
            )) if column == run.start
        ),
        "{:?}: the host refuses prepared stack {k} chain {i}'s opening, got {refusal:?}",
        base.hash
    );
    let g = prepared.group;
    let leaf = (0..plan.partition().num_leaves())
        .find(|&l| plan.partition().leaf(l).contains(&g))
        .expect("a leaf holds the group");
    let (program, marks) = onechain_steps::record(|| {
        leaf_program_with(&plan, leaf, LeafChecks::ALL).expect("the leaf emits")
    });
    let hasher = program.hasher(crate::hash_pin::BLOCK_HASHER);
    let honest = block_leaf_arena(&plan, &proof, leaf).expect("the honest arena");
    super::executor::execute_serial(&program, &honest, &hasher).expect("the honest leaf executes");
    let forged_arena = block_leaf_arena(&plan, &forged, leaf).expect("the forged arena");
    let error = super::executor::execute_serial(&program, &forged_arena, &hasher)
        .expect_err("the leaf refuses the forgery");
    assert_eq!(
        onechain_steps::failed_at(&program, &marks, &error),
        Some((
            Some(run.start),
            Step::Opening {
                round: 0,
                query: 1,
                tree
            }
        )),
        "{:?}: leaf {leaf} refuses prepared stack {k} chain {i} at tree {tree}'s opening, got \
         {error:?}",
        base.hash
    );
    println!(
        "ONECHAIN PREPARED {:?} stack 2^{stack}: REFUSED prepared stack {k} chain {i} (run {run:?}) \
         — host at its opening, leaf {leaf} of {} at tree {tree}'s opening at query 1",
        base.hash,
        plan.partition().num_leaves()
    );
}

/// The code review's M-g: a PREPARED stack opened by a GROUPED chain. Under a
/// stack of 2^8, DECODE's five ELF-derived columns (2^7 rows for this program)
/// stack into three polynomials — one chain of three, slot 3 absent — while
/// twelve polynomials a group keep each group's cells those of the 2^10
/// format. The grouped prepared chain is forged in tree 1.
#[test]
#[ignore = "proves small blocks and executes their leaves; box tier"]
fn a_grouped_prepared_stack_verifies_and_its_leaf_refuses_a_forgery() {
    for argue in [ArgueFormat::PerTable, ArgueFormat::BATCHED] {
        forged_prepared_chain(crate::hash_pin::BLOCK_BASE, argue, 8, 12, |chains| {
            chains.iter().copied().find(|&(_, _, width)| width >= 2)
        });
    }
}

/// The code review's M-g: a prepared stack SPLIT into two chains. Under a stack
/// of 2^7 each of DECODE's five 2^7-row columns is a polynomial of its own:
/// runs [4, 1]. The LAST chain — whose roots are the stack's derived roots from
/// polynomial 4 on — is forged, so each chain's slice of the derived roots
/// (`roots[run]`) is what refuses it, by position.
#[test]
#[ignore = "proves small blocks and executes their leaves; box tier"]
fn a_split_prepared_stack_refuses_a_forgery_in_its_last_chain() {
    for argue in [ArgueFormat::PerTable, ArgueFormat::BATCHED] {
        split_prepared_stack_refuses_a_forgery_in_its_last_chain(argue);
    }
}

fn split_prepared_stack_refuses_a_forgery_in_its_last_chain(argue: ArgueFormat) {
    forged_prepared_chain(crate::hash_pin::BLOCK_BASE, argue, 7, 24, |chains| {
        chains.iter().copied().rfind(|&(k, i, _)| {
            i >= 1
                && chains
                    .iter()
                    .any(|&(kk, ii, w)| kk == k && ii == 0 && w >= 2)
        })
    });
}
