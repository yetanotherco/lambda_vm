//! The no-epoch WHIR block's recursion ([`super::whir_block`]): the group
//! partition; the leaves over small block proofs, closing the bus between them;
//! a refusal per check a leaf or a node adds, each shown load-bearing by its
//! mutation; the tree to the top the plan derives; and the real block (box).

use crate::block_whir::{
    self, BlockFormat, BlockOptions, BlockWhirProof, prove_block_whir, verify_block_whir,
};
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
use super::per_table_aggregator_tests::{RealChild, child_arena_words, real_child_timed};
use super::proof::{LfmProof, aggregation_wrap_options, lfm_prove};
use super::whir_block::{
    BLOCK_FAN_IN, BlockPartition, LEAF_PERMS_CAP, LeafChecks, WhirBlockPlan, artifacts_of,
    block_leaf_arena, emit_share, group_arena_words, id_words, leaf_arena, leaf_partition,
    leaf_program_with, out_halves, partition_groups, verify_block_tree, verify_block_tree_under,
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
        pack_finished: true,
        gpack: true,
        spill: crate::block_whir::BlockSpillPolicy::Off,
        regen: None,
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
    execute(&program, &arenas, &crate::hash_pin::BLOCK_HASHER)
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
#[test]
#[ignore = "proves small blocks; box tier"]
fn the_groups_phase_b_hands_over_build_every_leafs_arena() {
    for (format, name) in [
        (small_format(), "all_instructions_64"),
        (small_batched(), "all_instructions_64"),
        (small_batched(), "test_keccak"),
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
    let block = [
        65825, 64356, 64405, 64474, 66559, 65146, 81574, 68482, 105991,
    ];
    let p = leaf_partition(&block, None, LEAF_PERMS_CAP).expect("partitions");
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

/// ★ Every leaf executes over an honest block proof, all of them publish the
/// same id, state and output — the plan's — and their shares cancel: the
/// block's bus, checked once over every table, closes across the leaves.
#[test]
#[ignore = "executes block leaves over small block proofs; box tier"]
fn whir_block_leaves_execute_and_close_the_bus() {
    let format = small_format();
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

/// The leaves' front draws the host verifier's challenges: `z, α, β` after the
/// statement and every root (the groups' and the derived prepared ones), and a
/// group fork's first draw after the group's index.
#[test]
#[ignore = "executes a block leaf's front over a small block proof; box tier"]
fn the_block_front_draws_the_hosts_challenges() {
    use crypto::fiat_shamir::default_transcript::DefaultTranscript;
    use crypto::fiat_shamir::is_transcript::IsTranscript;
    let format = small_format();
    let (elf, proof) = small_block("test_commit_4", &format);
    let plan = plan_of(&elf, &proof, &format, Some(1));
    let g = plan.num_groups() - 1;
    let program = super::whir_block::front_program(&plan, g);
    let arena: Vec<LfmWord> = proof
        .proof
        .roots
        .iter()
        .map(super::algebraic_commit::commitment_to_digest)
        .collect();
    let words: Vec<LfmWord> = execute(&program, &[arena], &crate::hash_pin::BLOCK_HASHER)
        .expect("the front executes")
        .public_words
        .iter()
        .map(|(_, w)| *w)
        .collect();
    let opts = ProofOptions::default_test_options();
    let frame = block_whir::block_frame(proof.statement(), &elf, &opts, &format).expect("frame");
    let derived: Vec<multilinear::whir_commit::Commitment> = plan
        .prepared()
        .iter()
        .flat_map(|p| p.roots.iter().copied())
        .collect();
    let host: Vec<FEE> = crate::with_whir_hash!(|H| {
        type T = DefaultTranscript<FEE_FIELD, <H as multilinear::whir_hash::WhirHash>::Transcript>;
        let mut t = T::new(&[]);
        block_whir::absorb_block(
            &mut t,
            &elf,
            &proof.public_output,
            &proof.table_counts,
            proof.num_private_input_pages,
            &proof.runtime_page_ranges,
            &proof.table_num_vars,
            &frame.config,
            &proof.groups,
        );
        stark::multilinear_table::absorb_roots::<FEE_FIELD, _>(
            &mut t,
            &proof.proof.roots,
            &derived,
        );
        let mut drawn: Vec<FEE> = (0..3).map(|_| t.sample_field_element()).collect();
        let mut fork = t.clone();
        fork.append_bytes(&(g as u64).to_le_bytes());
        drawn.push(fork.sample_field_element());
        drawn
    });
    let machine: Vec<FEE> = words.iter().map(ext_of).collect();
    assert_eq!(machine, host, "z, α, β and the fork's first draw");
}

/// ★ A leaf refuses a tampered witness — a group root, a table's argument, a
/// prepared opening — and the prepared opening is the check refusing the last:
/// with it left out, the tampered opening executes.
#[test]
#[ignore = "executes block leaves over small block proofs; box tier"]
fn a_block_leaf_refuses_a_tampered_witness() {
    let format = small_format();
    let (elf, proof) = small_block("test_commit_4", &format);
    let plan = plan_of(&elf, &proof, &format, Some(1));
    run_leaf(&plan, &proof, 0, LeafChecks::ALL).expect("the honest leaf executes");

    let mut root = proof.clone();
    root.proof.roots[0][0] ^= 1;
    assert!(
        run_leaf(&plan, &root, 0, LeafChecks::ALL).is_err(),
        "a tampered group root"
    );

    let mut table = proof.clone();
    table.proof.tables[0].bus_output.0 += FEE::one();
    assert!(
        run_leaf(&plan, &table, 0, LeafChecks::ALL).is_err(),
        "a tampered table argument"
    );

    let mut prepared = proof.clone();
    prepared.prepared[0].polys[0].final_value += FEE::one();
    assert!(
        run_leaf(&plan, &prepared, 0, LeafChecks::ALL).is_err(),
        "a tampered prepared opening"
    );
    let without = LeafChecks {
        prepared: false,
        ..LeafChecks::ALL
    };
    assert!(
        run_leaf(&plan, &prepared, 0, without).is_ok(),
        "with the prepared openings left out the tamper executes, so they are what refuses it"
    );
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
            pack_finished: true,
            gpack: true,
            spill: crate::block_whir::BlockSpillPolicy::Off,
            regen: None,
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
#[ignore = "proves the dense fixture under RPX and executes its leaf; box tier"]
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
    let derived = DerivedChild::from_artifacts(&artifacts, &opts, words)?;
    let t = std::time::Instant::now();
    let proof =
        lfm_prove(program, &artifacts, arenas, &opts).map_err(|e| format!("{label}: {e:?}"))?;
    let t_prove = t.elapsed().as_secs_f64();
    let (child, verify) = real_child_timed(artifacts, opts, &proof);
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

/// ★ G3 through the plan: test_commit_4's heaviest group costs ≈ 220 k
/// permutations; under a 200 k leaf cap the plan refuses the statement (no
/// leaf can hold that group), under the production cap it derives today's
/// partition.
#[test]
#[ignore = "proves a small block; box tier"]
fn a_group_over_the_leaf_cap_is_refused() {
    let format = small_format();
    let (elf, proof) = small_block("test_commit_4", &format);
    let opts = ProofOptions::default_test_options();
    let derive = |cap: usize| {
        WhirBlockPlan::derive_capped(
            &elf,
            &opts,
            &format,
            proof.statement(),
            None,
            BLOCK_FAN_IN,
            None,
            cap,
        )
    };
    let plan = derive(LEAF_PERMS_CAP).expect("the production cap derives the plan");
    let heaviest = *plan.costs().iter().max().expect("a group");
    assert!(
        heaviest > 200_000 && heaviest <= LEAF_PERMS_CAP,
        "{heaviest}"
    );
    assert_eq!(
        plan.partition().leaves(),
        plan_of(&elf, &proof, &format, None).partition().leaves()
    );
    let refused = derive(heaviest - 1);
    let msg = refused.as_ref().err().cloned().unwrap_or_default();
    assert!(
        msg.contains("costs") && msg.contains(&format!("at most {}", heaviest - 1)),
        "a group over the leaf cap must be refused, got {msg:?}"
    );
    println!(
        "WHIR BLOCK LEAF CAP: heaviest group {heaviest}; under a cap of {}: {msg}",
        heaviest - 1
    );
}

/// ★ ECDAS cut through the block's recursion: test_ecsm_multi with ECDAS in
/// 16-row tables (four; the 0xABCDEF call through three) proves as a block
/// whose tree the verifier accepts. The same statement with one ECDAS table
/// stated over [`block_whir::BLOCK_ECDAS_MAX_VARS`] is refused before any
/// program is derived.
#[test]
#[ignore = "proves block leaves and nodes over a small block; box tier"]
fn the_whir_block_tree_verifies_a_split_ecdas() {
    super::device_permit::arm(1);
    let format = small_format();
    let (elf, proof) = small_block_at("test_ecsm_multi", &format, 4);
    assert_eq!(proof.table_counts.ecdas, 4, "four ECDAS tables");
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
        "WHIR BLOCK TREE ECDAS SPLIT: {} groups, 4 ECDAS tables, accepted; an ECDAS over its cap \
         refused ({})",
        plan.num_groups(),
        refused.err().unwrap_or_default()
    );
}

/// ★ KECCAK and ECSM cut through the block's recursion: test_keccak_multi's
/// three permutations and test_ecsm_multi's three scalar multiplications, one
/// row a table (four tables each: three calls and the padding row), prove as
/// blocks whose trees the verifier derives and accepts. Each statement with
/// its first table stated over its cap ([`block_whir::BLOCK_KECCAK_MAX_VARS`],
/// [`block_whir::BLOCK_ECSM_MAX_VARS`]) is refused before any program is
/// derived.
#[test]
#[ignore = "proves two small blocks and their trees; box tier"]
fn the_whir_block_tree_verifies_split_keccak_and_ecsm() {
    super::device_permit::arm(1);
    let format = small_format();
    let opts = ProofOptions::default_test_options();
    let wrap = aggregation_wrap_options();
    for (guest, name) in [("test_keccak_multi", "KECCAK"), ("test_ecsm_multi", "ECSM")] {
        let (elf, proof) = small_block_cut(guest, &format, |o| match name {
            "KECCAK" => o.keccak_rows_log2 = 0,
            _ => o.ecsm_rows_log2 = 0,
        });
        let (_, range, cap) = block_whir::chunked_table_ranges(&proof.table_counts)
            .into_iter()
            .find(|(n, ..)| *n == name)
            .expect("the table's range");
        assert_eq!(range.len(), 4, "{name}: one row a table");
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
            "WHIR BLOCK TREE {name} SPLIT: {} groups, 4 {name} tables, accepted; a {name} over its \
             cap refused ({})",
            plan.num_groups(),
            refused.err().unwrap_or_default()
        );
    }
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
        super::proof::verify_against_artifacts(&own, &other.proof, &other.public_words, &wrap),
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
        let mut b = LfmBuilder::new().with_wrap_hash(super::edsl::WrapHash::production());
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
/// leaf, `N(..)` / `T(..)` a node over its children's). On a pool, a level's
/// node 0 ends its build only once node 1 has ended its own (two emitter
/// threads take them at once), so the level's builds finish out of order
/// whatever the host's load: a wait decides it, not a sleep. Returns every
/// slot's content and the order the builds finished in.
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
    // Which builds have ended, level by level, for node 0 to wait on node 1.
    let ended = std::sync::Mutex::new(
        shape
            .iter()
            .map(|l| vec![false; l.arities.len()])
            .collect::<Vec<_>>(),
    );
    let changed = std::sync::Condvar::new();
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
            // Bounded, so a scheduler that never runs node 1 beside node 0
            // fails the order assertion instead of hanging the test.
            if threads.is_some() && j == 0 && shape[lv].arities.len() > 1 {
                let t = std::time::Instant::now();
                let mut done = ended.lock().expect("the ends");
                while !done[lv][1] && t.elapsed() < std::time::Duration::from_secs(10) {
                    done = changed
                        .wait_timeout(done, std::time::Duration::from_millis(50))
                        .expect("the ends")
                        .0;
                }
            }
            finished.lock().expect("the log").push((lv, j));
            ended.lock().expect("the ends")[lv][j] = true;
            changed.notify_all();
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

/// ★ W3's readout on a real block (box only, `--ignored`, `--features cuda`,
/// `LAMBDA_VM_WHIR_HASH=rpx`), through the one driver the CLI's `prove-block`
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
/// when unset), `W3_SIBLINGS` and `W3_FAN_IN` ([`BLOCK_FAN_IN`] when unset).
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
            super::proof::verify_against_artifacts(artifacts, &lfm.proof, &lfm.public_words, &wrap),
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
    // The production verifier when the run is at its presets; the fixture form
    // only when a knob moved the tree off them.
    let verdict =
        if leaves.is_none() && fan_in == BLOCK_FAN_IN && argue == BlockFormat::production().argue {
            verify_block_tree(&elf, proof.statement(), top)
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
#[test]
#[ignore = "executes block leaves over small block proofs; box tier"]
fn batched_block_leaves_execute_and_close_the_bus() {
    let format = small_batched();
    for (name, leaves) in [
        ("all_instructions_64", Some(3)),
        ("test_commit_4", Some(2)),
        ("test_keccak", None),
        ("test_commit_4", Some(1)),
    ] {
        let (elf, proof) = small_block(name, &format);
        assert_eq!(proof.argues.len(), proof.groups.len());
        let plan = plan_of(&elf, &proof, &format, leaves);
        let per_table = plan_of(&elf, &proof, &small_format(), leaves);
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
/// leaf that executes; the plan at 20 emits one that refuses the same proof.
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
    let refusal = match block_leaf_arena(&strict, &proof, 0) {
        Err(e) => format!("the arena: {e}"),
        Ok(arenas) if program.arena_schema.lens != vec![arenas[0].len() as u32] => format!(
            "the arena's length: {} words for a leaf hinting {:?}",
            arenas[0].len(),
            program.arena_schema.lens
        ),
        Ok(arenas) => match execute(&program, &arenas, &crate::hash_pin::BLOCK_HASHER) {
            Err(e) => format!("the execution: {e:?}"),
            Ok(_) => panic!("a leaf at 20 bits executed over a block ground at 18"),
        },
    };
    println!("GRIND LEAF REFUSAL: {refusal}");
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
