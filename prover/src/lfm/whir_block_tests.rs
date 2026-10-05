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
use super::proof::{
    LfmFilled, LfmProof, aggregation_wrap_options, decide_lfm_residency, lfm_execute_and_fill,
    lfm_prove, take_prove_split,
};
use super::whir_block::{
    BLOCK_FAN_IN, BlockPartition, LEAF_PERMS_CAP, LeafChecks, WhirBlockPlan, artifacts_of,
    block_leaf_arena, emit_share, group_arena_words, id_words, leaf_arena, leaf_partition,
    leaf_program_with, out_halves, partition_groups, verify_block_tree, verify_block_tree_under,
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
        drop_levels: 3,
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
        rest_layout_bytes: Some(1 << 20),
        pack_finished: true,
        spill: crate::block_whir::BlockSpillPolicy::Off,
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
        options.drop_levels = 3;
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
            drop_levels: 3,
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
            rest_layout_bytes: Some(1 << 20),
            pack_finished: true,
            spill: crate::block_whir::BlockSpillPolicy::Off,
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

/// A node program built ahead of the proofs below it.
struct TreeNode {
    program: LfmProgram,
    artifacts: super::registry::LfmArtifacts,
    derived: DerivedChild,
    /// Seconds emitting it and building its artifacts.
    built: f64,
    /// When its builder finished it, in seconds since the tree started.
    built_at: f64,
}

/// One level's readout: its wall, and per program (artifacts, prove) seconds
/// and its [`ProgramTimes`].
struct LevelTiming {
    wall: f64,
    programs: Vec<(f64, f64)>,
    times: Vec<ProgramTimes>,
}

/// When one program of the tree ran, in seconds since the tree started, and
/// its prove's split ([`take_prove_split`]: execute, fill, the prove net of
/// the card wait, the card wait) — the `W3 TIMES` readout.
#[derive(Clone, Copy, Default)]
struct ProgramTimes {
    /// Its program (nodes) and artifacts built.
    built_at: f64,
    /// Its worker took it.
    start: f64,
    /// Its proof done.
    end: f64,
    split: Option<(f64, f64, f64, f64)>,
}

/// The split of the prove that just ran on this thread.
fn prove_split_now() -> Option<(f64, f64, f64, f64)> {
    take_prove_split().map(|s| (s.execute, s.fill, s.multi_prove, s.permit_wait))
}

/// One group's share of the proof, owned, as phase B hands it over
/// ([`block_whir::GroupObserver`]).
struct GroupMsg {
    group: usize,
    roots: Vec<multilinear::whir_commit::Commitment>,
    tables: Vec<stark::multilinear_table::TableProof<crate::tables::types::GoldilocksExtension>>,
    argue:
        Option<stark::multilinear_table::BatchedArgue<crate::tables::types::GoldilocksExtension>>,
    opening: multilinear::stacked_eval::StackedProof<
        crate::tables::types::GoldilocksField,
        crate::tables::types::GoldilocksExtension,
    >,
    prepared: Option<
        multilinear::stacked_eval::StackedProof<
            crate::tables::types::GoldilocksField,
            crate::tables::types::GoldilocksExtension,
        >,
    >,
    at: f64,
}

/// What the early leaf's thread hands back: the filled leaf, its arena (for
/// the off-the-clock check against the finished proof's), and when its
/// execute + fill ran.
type EarlyOut = (LfmFilled, Vec<Vec<LfmWord>>, f64, f64);

/// The first leaf whose groups phase B finished while another group was
/// still to come, executing and filling on a thread of its own.
struct EarlyLeaf {
    leaf: usize,
    /// When its last group's opening ended (seconds since the prove started).
    complete_at: f64,
    handle: std::thread::JoinHandle<Result<EarlyOut, String>>,
}

/// A value one thread publishes once and others wait for. A publisher that
/// unwinds before publishing leaves an error behind ([`PublishGuard`]), so no
/// waiter blocks on a value that will never come.
struct Published<T>(std::sync::OnceLock<Result<T, String>>);

impl<T> Published<T> {
    fn new() -> Self {
        Self(std::sync::OnceLock::new())
    }

    fn wait(&self) -> Result<&T, String> {
        self.0.wait().as_ref().map_err(Clone::clone)
    }

    fn take(self) -> Result<T, String> {
        self.0
            .into_inner()
            .unwrap_or_else(|| Err("never published".to_string()))
    }
}

/// Publishes an error on drop unless [`PublishGuard::publish`] ran first.
struct PublishGuard<'a, T>(&'a Published<T>);

impl<T> PublishGuard<'_, T> {
    fn publish(self, value: Result<T, String>) {
        let _ = self.0.0.set(value);
    }
}

impl<T> Drop for PublishGuard<'_, T> {
    fn drop(&mut self) {
        let _ = self.0.0.set(Err("the publisher unwound".to_string()));
    }
}

/// The leaves' artifacts as their builder publishes them: per leaf, its
/// artifacts, its derived shape and the seconds building them.
type LeafBuilt = Vec<(super::registry::LfmArtifacts, DerivedChild, f64)>;

/// Publishes an error into every slot still empty when it drops: a node
/// builder that stops early (an error, or a panic) leaves no level waiting
/// for a node that will never come.
struct FailUnpublished<'a, T>(&'a [Vec<Published<T>>]);

impl<T> Drop for FailUnpublished<'_, T> {
    fn drop(&mut self) {
        for slot in self.0.iter().flatten() {
            let _ = slot.0.set(Err("the node builder stopped".to_string()));
        }
    }
}

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

/// A tree's nodes built level by level from the leaves' shapes `leaves`, each
/// published to its slot (level, node) as soon as it is built: `emit` makes a
/// node's program from its level, index, children's shapes (in order) and
/// whether it is the top; `finish` builds the rest of the node from it (its
/// artifacts: it may take the card); `shape_of` gives a built node's shape for
/// the level above. A node depends on its children's shapes, never on their
/// proofs, so the whole tree can be built while the leaves prove.
///
/// With `pool`, a level's programs are emitted together on the pool's threads,
/// and this thread finishes each as its emission ends and publishes it into
/// ITS OWN slot, whatever order they finish in; without, emitted and finished
/// one after another here. `finish` always runs on this thread, never on a
/// rayon worker: a worker that waits inside rayon while it holds the card runs
/// queued jobs meanwhile, and a sibling that takes the card is a second hold on
/// one thread (BIG 569). A build that fails leaves its error in its slot, and
/// every slot still empty when this returns or unwinds gets one
/// ([`FailUnpublished`]).
#[allow(clippy::too_many_arguments)]
fn build_levels<C: Sync, P: Send, N: Send + Sync>(
    shape: &[super::per_table_aggregator::Level],
    leaves: &[&C],
    slots: &[Vec<Published<N>>],
    shape_of: impl Fn(&N) -> &C + Sync,
    emit: impl Fn(usize, usize, &[&C], bool) -> Result<P, String> + Sync,
    finish: impl Fn(usize, usize, P) -> Result<N, String>,
    pool: Option<&rayon::ThreadPool>,
) {
    let _fail = FailUnpublished(slots);
    for (lv, arities) in shape.iter().enumerate() {
        let top = lv + 1 == shape.len();
        let below: Vec<&C> = match lv {
            0 => leaves.to_vec(),
            _ => match slots[lv - 1]
                .iter()
                .map(|s| s.wait().map(&shape_of))
                .collect::<Result<Vec<_>, String>>()
            {
                Ok(below) => below,
                Err(_) => return,
            },
        };
        let mut at = 0usize;
        let groups: Vec<std::ops::Range<usize>> = arities
            .arities
            .iter()
            .map(|&a| {
                at += a;
                at - a..at
            })
            .collect();
        if groups.last().map_or(0, |g| g.end) != below.len() {
            let _ = slots[lv][0].0.set(Err(format!(
                "level {}: arities cover {at} of {} children",
                lv + 1,
                below.len()
            )));
            return;
        }
        match pool {
            Some(pool) => {
                let (tx, rx) = std::sync::mpsc::channel::<(usize, Result<P, String>)>();
                pool.in_place_scope(|scope| {
                    for (j, kids) in groups.iter().enumerate() {
                        let (tx, emit, kids) = (tx.clone(), &emit, &below[kids.clone()]);
                        scope.spawn(move |_| {
                            let _ = tx.send((j, emit(lv, j, kids, top)));
                        });
                    }
                    drop(tx);
                    // This thread, not a pool worker, finishes each node as
                    // its program arrives.
                    for (j, program) in rx {
                        let _ = slots[lv][j].0.set(program.and_then(|p| finish(lv, j, p)));
                    }
                });
            }
            None => {
                for (j, kids) in groups.iter().enumerate() {
                    let node =
                        emit(lv, j, &below[kids.clone()], top).and_then(|p| finish(lv, j, p));
                    let _ = slots[lv][j].0.set(node);
                }
            }
        }
    }
}

/// [`build_levels`] over a toy tree whose programs are their texts (`L3` a
/// leaf, `N(..)` / `T(..)` a node over its children's): each node's build
/// sleeps longer the earlier it is in its level, so on a pool the level's
/// nodes finish in reverse. Returns every slot's content and the order the
/// builds finished in.
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
    let finished = std::sync::Mutex::new(Vec::new());
    let pool = threads.map(|n| {
        rayon::ThreadPoolBuilder::new()
            .num_threads(n)
            .build()
            .expect("a pool")
    });
    build_levels(
        &shape,
        &leaf_refs,
        &slots,
        |node: &String| node,
        |lv, j, kids: &[&String], top| {
            let late = shape[lv].arities.len() - j;
            std::thread::sleep(std::time::Duration::from_millis(4 * late as u64));
            finished.lock().expect("the log").push((lv, j));
            if fail_at == Some((lv, j)) {
                return Err(format!("node ({lv}, {j}) does not build"));
            }
            let kids: Vec<&str> = kids.iter().map(|k| k.as_str()).collect();
            Ok(format!(
                "{}({})",
                if top { "T" } else { "N" },
                kids.join(",")
            ))
        },
        // The finish is where a node takes the card: never on a rayon worker.
        |lv, j, program: String| match rayon::current_thread_index() {
            None => Ok(program),
            Some(w) => Err(format!("node ({lv}, {j}) finished on rayon worker {w}")),
        },
        pool.as_ref(),
    );
    let got = slots
        .iter()
        .map(|level| level.iter().map(|s| s.0.get().cloned()).collect())
        .collect();
    (got, finished.into_inner().expect("the log"))
}

/// ★ The node pipe builds the serial builder's nodes, each in its own slot,
/// whatever order a level's emissions finish in, and finishes every node (where
/// it takes the card) off the rayon workers: the median's shape (23 leaves at
/// fan-in 3), on pools of 2 to 4 threads, where each level's emissions finish
/// out of order, and serially.
#[test]
fn the_node_pipe_builds_the_serial_builders_nodes_in_any_completion_order() {
    let arities = vec![vec![3, 3, 3, 3, 3, 3, 3, 2], vec![3, 3, 2], vec![3]];
    let group =
        |kids: &[String], top: bool| format!("{}({})", if top { "T" } else { "N" }, kids.join(","));
    let leaves: Vec<String> = (0..23).map(|k| format!("L{k}")).collect();
    let mut want: Vec<Vec<String>> = Vec::new();
    let mut below = leaves;
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
    let want: Vec<Vec<Option<Result<String, String>>>> = want
        .into_iter()
        .map(|l| l.into_iter().map(|p| Some(Ok(p))).collect())
        .collect();
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

/// The tree's node programs and artifacts ([`build_levels`] over the leaves'
/// derived shapes), each published to its slot as soon as it is built.
///
/// With `pool` (`W3_NODE_PIPE`, on by default) a level's programs are emitted
/// together on the pool's threads — the builder's own, not the global pool, on
/// which a prover's join could steal an emission and leave the card idle
/// (#1013's FAST 454) — and this thread builds each one's artifacts (holding
/// the card) as it arrives; each level proves as its own nodes arrive. Without
/// it, one after another on this thread (the control: the builder before,
/// whose whole tree level 1 then waits for — at the median 32 s of serial
/// builds holding level 0 open 10.7 s past its last proof, BIG 565 / 568).
#[allow(clippy::too_many_arguments)]
fn build_nodes(
    plan: &WhirBlockPlan,
    shape: &[super::per_table_aggregator::Level],
    leaves: &Published<LeafBuilt>,
    slots: &[Vec<Published<TreeNode>>],
    wrap: &crate::ProofOptions,
    words: usize,
    t_tree: std::time::Instant,
    pool: Option<&rayon::ThreadPool>,
) {
    let Ok(leaf_built) = leaves.wait() else {
        let _fail = FailUnpublished(slots);
        return;
    };
    let leaf_shapes: Vec<&DerivedChild> = leaf_built.iter().map(|(_, d, _)| d).collect();
    build_levels(
        shape,
        &leaf_shapes,
        slots,
        |node: &TreeNode| &node.derived,
        |_, _, kids: &[&DerivedChild], top| -> Result<(LfmProgram, f64), String> {
            let t = std::time::Instant::now();
            let program = plan.node_program(kids, top)?;
            Ok((program, t.elapsed().as_secs_f64()))
        },
        |_, _, (program, emitted): (LfmProgram, f64)| -> Result<TreeNode, String> {
            let t = std::time::Instant::now();
            let artifacts = artifacts_of(&program, wrap);
            let derived = DerivedChild::from_artifacts(&artifacts, wrap, words)?;
            Ok(TreeNode {
                program,
                artifacts,
                derived,
                built: emitted + t.elapsed().as_secs_f64(),
                built_at: t_tree.elapsed().as_secs_f64(),
            })
        },
        pool,
    );
}

/// A proved program of the tree, as [`prove_dataflow`] publishes it: its
/// proof, its harvest for the level above, its prove's seconds (artifacts
/// wait, execute, fill and prove), its build seconds (nodes) and its stamps.
struct ProvedProgram {
    lfm: LfmProof,
    child: RealChild,
    prove: f64,
    built: f64,
    times: ProgramTimes,
}

/// One program of the tree: its level (0 = the leaves) and its index there.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct TreeAt {
    lv: usize,
    j: usize,
}

/// Every program of a tree of `leaves` leaves laid out as `shape`, in
/// topological order: the leaves, then each node level in turn. Workers that
/// take programs in this order reach a node only after every program before it
/// — its children among them — has been taken, so the earliest unfinished
/// program waits on nothing untaken: no deadlock, at any number of workers.
fn tree_order(leaves: usize, shape: &[super::per_table_aggregator::Level]) -> Vec<TreeAt> {
    let mut order: Vec<TreeAt> = (0..leaves).map(|j| TreeAt { lv: 0, j }).collect();
    for (l, level) in shape.iter().enumerate() {
        order.extend((0..level.arities.len()).map(|j| TreeAt { lv: l + 1, j }));
    }
    order
}

/// Node `at`'s children, as indices into level `at.lv − 1`.
fn tree_children(
    shape: &[super::per_table_aggregator::Level],
    at: TreeAt,
) -> std::ops::Range<usize> {
    let arities = &shape[at.lv - 1].arities;
    let start: usize = arities[..at.j].iter().sum();
    start..start + arities[at.j]
}

/// Proves every program of a tree on `workers` threads, taking them in `order`
/// ([`tree_order`]) and publishing each result into its slot (`results[lv][j]`,
/// level 0 the leaves): `prove` gets the program and its children's slots in
/// order (none for a leaf), to wait for as it needs them ([`wait_all`]), so a
/// node whose children are done proves while the rest of the level below still
/// does — and one that streams can start on its first child. With
/// `barrier` (the control, `W3_DATAFLOW=0`), a node first waits for the whole
/// level below, as level-by-level proving did. A failed or panicking prove
/// leaves an error in its slot and so in every slot above it; every slot still
/// empty when this returns or unwinds gets one.
fn prove_dataflow<R: Send + Sync>(
    shape: &[super::per_table_aggregator::Level],
    order: &[TreeAt],
    results: &[Vec<Published<R>>],
    workers: usize,
    barrier: bool,
    prove: impl Fn(TreeAt, &[&Published<R>]) -> Result<R, String> + Sync,
) {
    use super::per_table_aggregator_tests::in_index_order;
    let _fail = FailUnpublished(results);
    in_index_order(order.len(), workers, |i| {
        let at = order[i];
        let kids = match at.lv {
            0 => Ok(Vec::new()),
            _ => {
                let below = &results[at.lv - 1];
                let all = match barrier {
                    true => below.iter().try_for_each(|s| s.wait().map(drop)),
                    false => Ok(()),
                };
                all.map(|()| {
                    tree_children(shape, at)
                        .map(|c| &below[c])
                        .collect::<Vec<_>>()
                })
            }
        };
        let result = kids.and_then(|kids| {
            std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| prove(at, &kids)))
                .unwrap_or_else(|payload| {
                    let why = payload
                        .downcast_ref::<String>()
                        .cloned()
                        .or_else(|| payload.downcast_ref::<&str>().map(|s| s.to_string()))
                        .unwrap_or_default();
                    Err(format!("level {} program {}: panicked: {why}", at.lv, at.j))
                })
        });
        let _ = results[at.lv][at.j].0.set(result);
    });
}

/// The top node streamed: its execution starts on the first of its children to
/// be proved and runs each child's share as that child lands
/// ([`super::executor::StreamedExecution`]: the witness is the unstreamed
/// one's, word for word), so only the last child's share, the fill and the
/// prove remain after the last child. A node's arenas are its children's, in
/// order and the same number each, so child `c` is group `c`. The execute the
/// split reports is the part after the last child landed.
fn prove_streamed(
    node: &TreeNode,
    kids: &[&Published<ProvedProgram>],
    wrap: &crate::ProofOptions,
) -> Result<LfmProof, String> {
    use super::executor::StreamedExecution;
    let k = kids.len();
    let arenas = node.program.arena_schema.lens.len();
    if k == 0 || !arenas.is_multiple_of(k) {
        return Err(format!("{arenas} arenas over {k} children"));
    }
    let per = arenas / k;
    let hasher = node.artifacts.hasher;
    let held: Vec<std::cell::OnceCell<Vec<Vec<LfmWord>>>> =
        (0..k).map(|_| std::cell::OnceCell::new()).collect();
    let mut ex = StreamedExecution::new(
        &node.program,
        (0..arenas).map(|a| a / per).collect(),
        &hasher,
    )
    .map_err(|e| format!("{e:?}"))?;
    let mut last = std::time::Instant::now();
    in_arrival_order(kids, |c, kid| {
        let words = held[c].get_or_init(|| child_arena_words(&kid.child));
        last = std::time::Instant::now();
        ex.land(c, words)
            .map_err(|e| format!("child {c} lands: {e:?}"))
    })?;
    let execution = ex.finish().map_err(|e| format!("{e:?}"))?;
    let filled = super::proof::lfm_fill_executed(
        &node.program,
        execution,
        hasher,
        last.elapsed().as_secs_f64(),
    );
    filled
        .prove(&node.artifacts, wrap, decide_lfm_residency())
        .map_err(|e| format!("{e:?}"))
}

/// Whether the top streams: only if a child is still unproved when the top's
/// program and artifacts are ready. With every child already proved there is
/// nothing to stream behind, and the streamed path (its forward pass, then a
/// wave per child) costs more than one whole execution: at 1× the top's slot
/// fills after its last child, and streaming it anyway cost 0.19 s (BIG 617 /
/// 618). A failed child counts as done: the whole path reports its error.
fn top_streams<R>(kids: &[&Published<R>]) -> bool {
    kids.iter().any(|k| k.0.get().is_none())
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

/// Calls `land` on each child's result as it is published, in the order they
/// arrive, each exactly once; a child that failed fails this with its error.
fn in_arrival_order<R>(
    kids: &[&Published<R>],
    mut land: impl FnMut(usize, &R) -> Result<(), String>,
) -> Result<(), String> {
    let mut landed = vec![false; kids.len()];
    while landed.contains(&false) {
        let mut moved = false;
        for (c, kid) in kids.iter().enumerate() {
            if landed[c] {
                continue;
            }
            match kid.0.get() {
                None => {}
                Some(Err(e)) => return Err(e.clone()),
                Some(Ok(r)) => {
                    land(c, r)?;
                    landed[c] = true;
                    moved = true;
                }
            }
        }
        if !moved {
            std::thread::sleep(std::time::Duration::from_millis(1));
        }
    }
    Ok(())
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

/// Every child's result, each waited for in order.
fn wait_all<'r, R>(kids: &[&'r Published<R>]) -> Result<Vec<&'r R>, String> {
    kids.iter().map(|k| k.wait()).collect()
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

/// ★ Dataflow over the median's shape proves every program of the serial
/// level-by-level order, each in its own slot, whatever order they finish in
/// (each level in reverse), at 1 to 4 workers: and with two or more workers a
/// node proves while the level below is still proving, which the barrier
/// (the control) never lets it.
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
            // Does any node start before its level below has finished?
            let early = (1..got.len()).any(|lv| {
                let below_end = got[lv - 1]
                    .iter()
                    .map(|r| r.as_ref().expect("proved").2)
                    .max()
                    .unwrap_or(0);
                got[lv]
                    .iter()
                    .any(|r| r.as_ref().expect("proved").1 < below_end)
            });
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

/// The tree proved the way a prover would run it, with the leaves' programs
/// emitted beforehand (`leaves`, while phase B ran):
/// 1. the leaves' artifacts, one after another on a thread of their own (each
///    holds the card);
/// 2. the nodes' programs and artifacts — functions of the leaves' artifacts,
///    not of any proof — on a thread of their own ([`build_nodes`]), each
///    published to its slot as it is built, beside
/// 3. every program proved `siblings` at a time, leaves first, each harvested
///    without its verify ([`prove_dataflow`]): with `beside`, each leaf
///    executes and fills its traces while the artifacts are built (neither
///    reads them) and waits for them only to prove; without, it waits for every
///    artifact first (the order before, `W3_EXEC_BESIDE_ARTIFACTS=0`); each
///    node as soon as its own children are proved and its program is in its
///    slot — with `dataflow` off (the control), once its whole level below is
///    proved; without `node_pipe` (that control), once the builder has built
///    the whole tree.
///
/// `node_pipe` is the builder's own pool ([`build_nodes`]); `early`, when
/// given, is a leaf that already executed and filled while phase B ran
/// ([`EarlyLeaf`]): its worker joins it instead.
///
/// Returns each level's timing and every proof with its artifacts, level by
/// level, the top last — for the harness to verify off the clock — and the
/// early leaf's arena, if any.
#[allow(clippy::type_complexity, clippy::too_many_arguments)]
fn prove_tree_pipelined(
    plan: &WhirBlockPlan,
    proof: &BlockWhirProof,
    leaves: Vec<std::sync::Arc<LfmProgram>>,
    siblings: usize,
    beside: bool,
    early: Option<EarlyLeaf>,
    node_pipe: Option<&rayon::ThreadPool>,
    dataflow: bool,
    stream_top: Option<&std::sync::OnceLock<bool>>,
) -> Result<
    (
        Vec<LevelTiming>,
        Vec<(super::registry::LfmArtifacts, LfmProof)>,
        Option<(usize, Vec<Vec<LfmWord>>, f64, f64, f64)>,
    ),
    String,
> {
    use super::per_table_aggregator_tests::harvest_child;
    let early = std::sync::Mutex::new(early);
    let early_out: std::sync::Mutex<Option<(usize, Vec<Vec<LfmWord>>, f64, f64, f64)>> =
        std::sync::Mutex::new(None);
    let wrap = aggregation_wrap_options();
    let words = plan.child_layout().total();
    let t_tree = std::time::Instant::now();
    let shape = plan.levels();
    let mut timings = Vec::with_capacity(shape.len() + 1);
    let mut proofs = Vec::new();
    let built: Published<LeafBuilt> = Published::new();
    let leaf_built_at: Vec<std::sync::Mutex<f64>> =
        leaves.iter().map(|_| std::sync::Mutex::new(0.0)).collect();
    let slots: Vec<Vec<Published<TreeNode>>> = shape
        .iter()
        .map(|l| l.arities.iter().map(|_| Published::new()).collect())
        .collect();

    let order = tree_order(leaves.len(), &shape);
    let results: Vec<Vec<Published<ProvedProgram>>> = std::iter::once(leaves.len())
        .chain(shape.iter().map(|l| l.arities.len()))
        .map(|n| (0..n).map(|_| Published::new()).collect())
        .collect();

    let n_leaves = std::thread::scope(|outer| -> Result<usize, String> {
        // 2. the nodes' programs and artifacts, into their slots.
        let builder = outer.spawn(|| {
            build_nodes(
                plan, &shape, &built, &slots, &wrap, words, t_tree, node_pipe,
            )
        });
        std::thread::scope(|scope| {
            // 1. the leaves' artifacts, one after another on this thread: each
            // build holds the card throughout (armed), so building them as
            // rayon jobs bought no overlap, and a holder on a rayon worker can
            // run a sibling build that takes the card again (BIG 569).
            scope.spawn(|| {
                let guard = PublishGuard(&built);
                guard.publish(
                    leaves
                        .iter()
                        .zip(&leaf_built_at)
                        .map(|(program, at)| -> Result<_, String> {
                            let t = std::time::Instant::now();
                            let artifacts = artifacts_of(program, &wrap);
                            let derived = DerivedChild::from_artifacts(&artifacts, &wrap, words)?;
                            *at.lock().map_err(|_| "a built-at stamp is poisoned")? =
                                t_tree.elapsed().as_secs_f64();
                            Ok((artifacts, derived, t.elapsed().as_secs_f64()))
                        })
                        .collect::<Result<_, String>>(),
                );
            });
            // 3. every program of the tree, `siblings` at a time, leaves first;
            // each node as soon as its children are proved and its program is
            // in its slot ([`prove_dataflow`]).
            prove_dataflow(
                &shape,
                &order,
                &results,
                siblings,
                !dataflow,
                |at, kids: &[&Published<ProvedProgram>]| -> Result<ProvedProgram, String> {
                    let start = t_tree.elapsed().as_secs_f64();
                    if at.lv == 0 {
                        let k = at.j;
                        let ahead = {
                            let mut slot =
                                early.lock().map_err(|_| "the early slot is poisoned")?;
                            if slot.as_ref().is_some_and(|e| e.leaf == k) {
                                slot.take()
                            } else {
                                None
                            }
                        };
                        let t = std::time::Instant::now();
                        let filled = match ahead {
                            Some(e) => {
                                let (filled, arena, started, secs) =
                                    e.handle.join().map_err(|_| {
                                        format!("leaf {k}: the early executor panicked")
                                    })??;
                                *early_out
                                    .lock()
                                    .map_err(|_| "the early readout is poisoned")? =
                                    Some((k, arena, e.complete_at, started, secs));
                                filled
                            }
                            None => {
                                let arenas = block_leaf_arena(plan, proof, k)?;
                                if !beside {
                                    built.wait()?;
                                }
                                lfm_execute_and_fill(
                                    &leaves[k],
                                    &arenas,
                                    crate::hash_pin::BLOCK_HASHER,
                                )
                                .map_err(|e| format!("leaf {k}: {e:?}"))?
                            }
                        };
                        let artifacts = &built.wait()?[k].0;
                        let lfm = filled
                            .prove(artifacts, &wrap, decide_lfm_residency())
                            .map_err(|e| format!("leaf {k}: {e:?}"))?;
                        let prove = t.elapsed().as_secs_f64();
                        let times = ProgramTimes {
                            built_at: 0.0,
                            start,
                            end: t_tree.elapsed().as_secs_f64(),
                            split: prove_split_now(),
                        };
                        let child = harvest_child(artifacts.clone(), wrap.clone(), &lfm);
                        return Ok(ProvedProgram {
                            lfm,
                            child,
                            prove,
                            built: 0.0,
                            times,
                        });
                    }
                    // The control of the node pipeline: no node proves before
                    // the whole tree is built, as when the builder joined level 0.
                    if node_pipe.is_none() {
                        for slot in slots.iter().flatten() {
                            slot.wait()?;
                        }
                    }
                    let node = slots[at.lv - 1][at.j].wait()?;
                    let t = std::time::Instant::now();
                    // The top streams only if a child is still unproved now
                    // that its program and artifacts are here; the choice is
                    // kept for the readout.
                    let streamed = match stream_top {
                        Some(chose) if at.lv == shape.len() => {
                            let streams = top_streams(kids);
                            let _ = chose.set(streams);
                            streams
                        }
                        _ => false,
                    };
                    let lfm = if streamed {
                        prove_streamed(node, kids, &wrap)
                            .map_err(|e| format!("the top, streamed: {e}"))?
                    } else {
                        let kids = wait_all(kids)?;
                        let arenas: Vec<Vec<LfmWord>> = kids
                            .iter()
                            .flat_map(|k| child_arena_words(&k.child))
                            .collect();
                        lfm_prove(&node.program, &node.artifacts, &arenas, &wrap)
                            .map_err(|e| format!("level {} node {}: {e:?}", at.lv, at.j))?
                    };
                    let prove = t.elapsed().as_secs_f64();
                    let times = ProgramTimes {
                        built_at: node.built_at,
                        start,
                        end: t_tree.elapsed().as_secs_f64(),
                        split: prove_split_now(),
                    };
                    let child = harvest_child(node.artifacts.clone(), wrap.clone(), &lfm);
                    Ok(ProvedProgram {
                        lfm,
                        child,
                        prove,
                        built: node.built,
                        times,
                    })
                },
            );
        });
        let leaf_built = built.wait()?;
        builder
            .join()
            .map_err(|_| "the node builder panicked".to_string())?;
        Ok(leaf_built.len())
    })?;
    // Each level's readout: a level's wall is from the level below's last
    // proof to its own (level 0: from the tree's start), now that a node may
    // prove before its level below is done.
    let leaf_secs: Vec<f64> = built.wait()?.iter().map(|(_, _, s)| *s).collect();
    let mut level_proofs: Vec<Vec<LfmProof>> = Vec::with_capacity(results.len());
    let mut below_end = 0.0f64;
    for (lv, level) in results.into_iter().enumerate() {
        let proved: Vec<ProvedProgram> = level
            .into_iter()
            .map(Published::take)
            .collect::<Result<_, String>>()?;
        let mut times = Vec::with_capacity(proved.len());
        let mut programs = Vec::with_capacity(proved.len());
        for (j, p) in proved.iter().enumerate() {
            let built = match lv {
                0 => leaf_secs[j],
                _ => p.built,
            };
            let built_at = match lv {
                0 => *leaf_built_at[j]
                    .lock()
                    .map_err(|_| "a built-at stamp is poisoned")?,
                _ => p.times.built_at,
            };
            times.push(ProgramTimes {
                built_at,
                ..p.times
            });
            programs.push((built, p.prove));
        }
        let end = proved.iter().map(|p| p.times.end).fold(0.0, f64::max);
        timings.push(LevelTiming {
            wall: end - below_end,
            programs,
            times,
        });
        below_end = end;
        level_proofs.push(proved.into_iter().map(|p| p.lfm).collect());
    }
    if level_proofs.last().map(Vec::len) != Some(1) {
        return Err(format!(
            "the tree closes to {:?} nodes",
            level_proofs.last().map(Vec::len)
        ));
    }
    // Off the clock: every proof with its artifacts, level by level.
    let mut level_proofs = level_proofs.into_iter();
    let leaf_lfms = level_proofs.next().unwrap_or_default();
    debug_assert_eq!(leaf_lfms.len(), n_leaves);
    for ((artifacts, _, _), lfm) in built.take()?.into_iter().zip(leaf_lfms) {
        proofs.push((artifacts, lfm));
    }
    for (level, lfms) in slots.into_iter().zip(level_proofs) {
        for (slot, lfm) in level.into_iter().zip(lfms) {
            proofs.push((slot.take()?.artifacts, lfm));
        }
    }
    let early_out = early_out
        .into_inner()
        .map_err(|_| "the early readout is poisoned")?;
    Ok((timings, proofs, early_out))
}

/// ★ W3's readout on a real block (box only, `--ignored`, `--features cuda`,
/// `LAMBDA_VM_WHIR_HASH=rpx`), run the way a prover would:
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
    let knob = |name: &str| -> Option<usize> {
        std::env::var(name).ok().map(|v| {
            v.trim()
                .parse()
                .unwrap_or_else(|_| panic!("{name} is a count"))
        })
    };
    let elf = std::fs::read(std::env::var("BLOCK_WHIR_ELF").expect("BLOCK_WHIR_ELF"))
        .expect("read the ELF");
    let input = std::fs::read(std::env::var("BLOCK_WHIR_INPUT").expect("BLOCK_WHIR_INPUT"))
        .expect("read the input");
    let leaves = knob("W3_LEAVES");
    let siblings = knob("W3_SIBLINGS").unwrap_or(3);
    let fan_in = knob("W3_FAN_IN").unwrap_or(BLOCK_FAN_IN);
    // `W3_EXEC_BESIDE_ARTIFACTS=0|1` (default 1): the leaves execute and fill
    // while their artifacts are built, or (0, the control) after all of them.
    let beside = knob("W3_EXEC_BESIDE_ARTIFACTS").is_none_or(|v| v != 0);
    // `W3_LEAF_DURING_PHASE_B=0|1` (default 1): the first leaf whose groups
    // phase B finished while another group was still to come executes and
    // fills right then, or (0, the control) with the others after the base.
    let early_on = knob("W3_LEAF_DURING_PHASE_B").is_none_or(|v| v != 0);
    // `BLOCK_WHIR_ARGUE=batched|per-table` (production: batched): each group's
    // tables argued together, at the format's bin cap (`BLOCK_WHIR_ARGUE_CAP=k`
    // for 2^k), or each on its own.
    let argue = match std::env::var("BLOCK_WHIR_ARGUE").as_deref().map(str::trim) {
        Ok("batched") => match knob("BLOCK_WHIR_ARGUE_CAP") {
            Some(cap) => ArgueFormat::Batched {
                bin_log_cells: u8::try_from(cap).expect("a cap below 2^256"),
            },
            None => ArgueFormat::BATCHED,
        },
        Ok("per-table") => ArgueFormat::PerTable,
        Err(_) => BlockFormat::production().argue,
        Ok(other) => panic!("BLOCK_WHIR_ARGUE={other}: per-table or batched"),
    };
    let format = BlockFormat {
        argue,
        ..BlockFormat::production()
    };
    println!("W3 ARGUE: {argue:?}");
    let mut options = BlockOptions::production();
    // `BLOCK_WHIR_STREAM_KECCAK_RND=1`: KECCAK_RND's chunks streamed (off by
    // default: NO EFFECT on this block, FAST 418).
    options.stream_keccak_rnd =
        std::env::var("BLOCK_WHIR_STREAM_KECCAK_RND").is_ok_and(|v| v.trim() == "1");
    // `BLOCK_WHIR_STREAM_MEMW_LT=1`: the MEMW-derived LT ops streamed per window.
    options.stream_memw_lt =
        std::env::var("BLOCK_WHIR_STREAM_MEMW_LT").is_ok_and(|v| v.trim() == "1");
    // `BLOCK_WHIR_LAYOUT_WORKERS=n`: the streamed chunks laid out on n threads
    // (production 3; 0 is the inline layout).
    if let Some(n) = knob("BLOCK_WHIR_LAYOUT_WORKERS") {
        options.layout_workers = n;
    }
    // `BLOCK_WHIR_LAYOUT_AHEAD=k` bounds the chunks unpacked at k + 1;
    // `none` lifts the bound (the reverted version, BIG 390).
    match std::env::var("BLOCK_WHIR_LAYOUT_AHEAD")
        .as_deref()
        .map(str::trim)
    {
        Ok("none") => options.layout_ahead = None,
        Ok(k) => {
            if let Ok(k) = k.parse() {
                options.layout_ahead = Some(k);
            }
        }
        Err(_) => {}
    }
    // `BLOCK_WHIR_PACK_REST=0|1` (production 0): the rest packed as it is laid
    // out.
    match std::env::var("BLOCK_WHIR_PACK_REST")
        .as_deref()
        .map(str::trim)
    {
        Ok("0") => options.pack_rest_as_laid_out = false,
        Ok("1") => options.pack_rest_as_laid_out = true,
        Ok(other) => panic!("BLOCK_WHIR_PACK_REST={other}: 0 or 1"),
        Err(_) => {}
    }
    // `BLOCK_WHIR_DROP_OPS=0`: the builder keeps the streamed chunks' ops
    // (production drops them).
    match std::env::var("BLOCK_WHIR_DROP_OPS")
        .as_deref()
        .map(str::trim)
    {
        Ok("0") => options.drop_streamed_ops = false,
        Ok("1") => options.drop_streamed_ops = true,
        _ => {}
    }
    // `BLOCK_WHIR_NARROW=wide|card|host` (production card): how phase A holds
    // each group's columns once committed.
    if let Some(narrow) = crate::block_whir::narrow_from_env() {
        options.narrow = narrow;
    }
    // `BLOCK_WHIR_KECCAK_LOG2=k`, `BLOCK_WHIR_ECSM_LOG2=k`: force a KECCAK or
    // ECSM split (production 2^18 / 2^17).
    crate::block_whir::chunk_cuts_from_env(&mut options);
    // `BLOCK_WHIR_UPLOAD_AHEAD=0|1` (production 1): phase A puts each group's
    // columns on the card beside the previous group's commit.
    if let Some(ahead) = crate::block_whir::upload_ahead_from_env() {
        options.upload_ahead = ahead;
    }
    println!("BLOCK UPLOAD AHEAD: {}", options.upload_ahead);
    // `BLOCK_WHIR_REST_LAYOUT=all|<MiB>` (production 2048),
    // `BLOCK_WHIR_KR_FINISH_CHUNKS=0|1` and `BLOCK_WHIR_PACK_FINISHED=0|1`
    // (production 1): the rest's layout in waves, KECCAK_RND built as its
    // tables, the finish's tables packed as they are built.
    crate::block_whir::rest_layout_from_env(&mut options);
    println!(
        "BLOCK REST LAYOUT CONFIG: waves of {} · KECCAK_RND built as its tables {} · finish packed {}",
        options
            .rest_layout_bytes
            .map_or("all at once".to_string(), |b| format!("{} MiB", b >> 20)),
        options.finish_keccak_rnd_chunks,
        options.pack_finished,
    );
    let opts = super::proof::block_base_options();
    let wrap = aggregation_wrap_options();
    super::device_permit::arm(siblings);
    println!(
        "W3 CONFIG: leaves {leaves:?} · siblings {siblings} · fan-in {fan_in} · KECCAK_RND streamed {} · MEMW LT streamed {} · layout workers {} (ahead {:?}) · rest packed as laid out {} · streamed ops dropped {}",
        options.stream_keccak_rnd,
        options.stream_memw_lt,
        options.layout_workers,
        options.layout_ahead,
        options.pack_rest_as_laid_out,
        options.drop_streamed_ops
    );

    let t0 = std::time::Instant::now();
    // The wall clock at the prove's start, to place a box's memory samples.
    println!(
        "W3 PROVE START: unix {:.3}",
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map_or(0.0, |d| d.as_secs_f64())
    );
    let sender = std::sync::Mutex::new(None::<std::sync::mpsc::Sender<_>>);
    let (tx, rx) = std::sync::mpsc::channel::<(
        block_whir::OwnedBlockStatement,
        Vec<block_whir::PreparedRoots>,
    )>();
    *sender.lock().expect("lock") = Some(tx);
    let observe = |statement: block_whir::BlockStatement<'_>,
                   roots: &[block_whir::PreparedRoots]| {
        if let Some(tx) = sender.lock().expect("lock").as_ref() {
            let _ = tx.send((statement.to_owned(), roots.to_vec()));
        }
    };
    // Each group's share of the proof as its opening ends, for the early leaf.
    let group_sender = std::sync::Mutex::new(None::<std::sync::mpsc::Sender<GroupMsg>>);
    let (gtx, grx) = std::sync::mpsc::channel::<GroupMsg>();
    // Without the early leaf no sender is kept, so the planner's loop ends
    // at once.
    *group_sender.lock().expect("lock") = early_on.then_some(gtx);
    let on_group = |opened: stark::multilinear_block::GroupOpened<'_, _, _>| {
        if let Some(tx) = group_sender.lock().expect("lock").as_ref() {
            let _ = tx.send(GroupMsg {
                group: opened.group,
                roots: opened.roots.to_vec(),
                tables: opened.tables.to_vec(),
                argue: opened.argue.cloned(),
                opening: opened.opening.clone(),
                prepared: opened.prepared.cloned(),
                at: t0.elapsed().as_secs_f64(),
            });
        }
    };
    let (elf_ref, opts_ref, format_ref) = (&elf, &opts, &format);
    #[allow(clippy::type_complexity)]
    let (proved, pre) = std::thread::scope(|scope| {
        let pre = scope.spawn(
            move || -> Result<
                (
                    WhirBlockPlan,
                    Vec<std::sync::Arc<LfmProgram>>,
                    f64,
                    f64,
                    Option<EarlyLeaf>,
                ),
                String,
            > {
                let (statement, roots) = rx
                    .recv()
                    .map_err(|_| "the prover never stated".to_string())?;
                let at = t0.elapsed().as_secs_f64();
                let plan = WhirBlockPlan::derive_with(
                    elf_ref,
                    opts_ref,
                    format_ref,
                    statement.view(),
                    leaves,
                    fan_in,
                    Some(&roots),
                )?;
                // One plain thread a leaf, off the rayon pool phase B is using.
                let programs: Vec<LfmProgram> = std::thread::scope(|inner| {
                    let handles: Vec<_> = (0..plan.partition().num_leaves())
                        .map(|k| {
                            let plan = &plan;
                            inner.spawn(move || plan.leaf_program(k))
                        })
                        .collect();
                    handles
                        .into_iter()
                        .map(|h| {
                            h.join()
                                .map_err(|_| "a leaf emitter panicked".to_string())?
                        })
                        .collect::<Result<_, String>>()
                })?;
                let programs: Vec<std::sync::Arc<LfmProgram>> =
                    programs.into_iter().map(std::sync::Arc::new).collect();
                let ready = t0.elapsed().as_secs_f64();
                // The groups as phase B finishes them (none unless `early_on`):
                // the first leaf complete while a group is still to come runs
                // its execute + fill now, on a thread of its own.
                let mut early = None;
                let mut words: Vec<Option<Vec<LfmWord>>> =
                    (0..plan.num_groups()).map(|_| None).collect();
                let mut block_roots = Vec::new();
                let mut done = 0usize;
                while let Ok(msg) = grx.recv() {
                    done += 1;
                    let g = msg.group;
                    let slot = words.get_mut(g).ok_or("a group the plan does not hold")?;
                    *slot = Some(group_arena_words(
                        &plan,
                        g,
                        &msg.tables,
                        msg.argue.as_ref(),
                        &msg.opening,
                        msg.prepared.as_ref(),
                    )?);
                    if block_roots.is_empty() {
                        block_roots = msg.roots;
                    }
                    if early.is_some() || done >= plan.num_groups() {
                        continue;
                    }
                    let partition = plan.partition();
                    let complete = (0..partition.num_leaves())
                        .find(|&k| partition.leaf(k).iter().all(|&g| words[g].is_some()));
                    if let Some(k) = complete {
                        let groups: Vec<Vec<LfmWord>> = partition
                            .leaf(k)
                            .iter()
                            .map(|&g| words[g].clone().unwrap_or_default())
                            .collect();
                        let arena = leaf_arena(&block_roots, groups);
                        let program = std::sync::Arc::clone(&programs[k]);
                        let handle = std::thread::spawn(move || -> Result<EarlyOut, String> {
                            let started = t0.elapsed().as_secs_f64();
                            let t = std::time::Instant::now();
                            let filled = lfm_execute_and_fill(
                                &program,
                                &arena,
                                crate::hash_pin::BLOCK_HASHER,
                            )
                            .map_err(|e| format!("early leaf {k}: {e:?}"))?;
                            Ok((filled, arena, started, t.elapsed().as_secs_f64()))
                        });
                        early = Some(EarlyLeaf {
                            leaf: k,
                            complete_at: msg.at,
                            handle,
                        });
                    }
                }
                Ok((plan, programs, at, ready, early))
            },
        );
        let proved = block_whir::prove_block_whir_observed_groups(
            &elf, &input, &opts, &format, &options, &observe, &on_group,
        );
        // A prover that failed before stating must not leave the planner waiting.
        *sender.lock().expect("lock") = None;
        *group_sender.lock().expect("lock") = None;
        (proved, pre.join())
    });
    let (proof, stamps) = proved.expect("the block proves");
    let base = t0.elapsed().as_secs_f64();
    let (plan, programs, stated_at, ready_at, early) = pre
        .expect("the planner did not panic")
        .expect("the plan and the leaves derive");
    print!("{}", stamps.report());
    // LT's chunk heights (what its per-chunk deduplication leaves): equal across
    // runs of one setting when the chunking is deterministic.
    {
        let frame = block_whir::block_frame(proof.statement(), &elf, &opts, &format)
            .expect("the statement's frame");
        let refs = frame.airs.air_refs();
        let lt: Vec<u8> = refs
            .iter()
            .zip(&proof.table_num_vars)
            .filter(|(air, _)| air.name().starts_with("LT["))
            .map(|(_, &n)| n)
            .collect();
        println!("W3 LT HEIGHTS: {lt:?}");
        println!(
            "W3 CHUNKED: {} (cuts KECCAK 2^{} · ECSM 2^{})",
            block_whir::chunked_census(&proof.table_counts, &proof.table_num_vars),
            options.keccak_rows_log2,
            options.ecsm_rows_log2,
        );
        // Each group's tables by AIR, in the group's order: which tables the
        // card waited for, and where the layout's chunks went.
        for (g, group) in proof.groups.iter().enumerate() {
            let mut kinds: Vec<(String, usize)> = Vec::new();
            for &index in group {
                let name = refs[index as usize].name();
                let kind = name.split('[').next().unwrap_or(name).to_string();
                match kinds.last_mut() {
                    Some((last, n)) if *last == kind => *n += 1,
                    _ => kinds.push((kind, 1)),
                }
            }
            let kinds: Vec<String> = kinds
                .into_iter()
                .map(|(kind, n)| if n == 1 { kind } else { format!("{kind}×{n}") })
                .collect();
            println!("W3 GROUP TABLES {g}: {}", kinds.join(" "));
        }
    }
    println!(
        "W3 BASE: {base:.2}s · statement at {stated_at:.2}s · plan + {} leaves emitted by {ready_at:.2}s ({})",
        programs.len(),
        if ready_at <= base {
            "inside the base"
        } else {
            "after the base"
        }
    );
    println!(
        "W3 PLAN: {} groups · costs {:?} (Σ {}) · {} leaves {:?} · {} prepared",
        plan.num_groups(),
        plan.costs(),
        plan.costs().iter().sum::<usize>(),
        plan.partition().num_leaves(),
        plan.partition().leaves(),
        plan.prepared().len()
    );
    // Each leaf's chips, real / padded rows: how far each sits from its next
    // doubling, and which one doubled when the in-guest work moves.
    for (k, program) in programs.iter().enumerate() {
        let chips: Vec<String> =
            super::airs::lfm_chip_census_with_hasher(program, crate::hash_pin::BLOCK_HASHER)
                .iter()
                .filter(|c| c.real_rows > 0)
                .map(|c| format!("{}={}/{}", c.name, c.real_rows, c.rows))
                .collect();
        println!("W3 LEAF CENSUS {k}: {}", chips.join(" "));
    }

    // The readouts above run on the clock, between the base and the tree: a
    // prover prints none of them, so the whole block is also given without.
    let readouts = t0.elapsed().as_secs_f64() - base;
    // `W3_NODE_PIPE=0|1` (default 1): the nodes' programs and artifacts built
    // on a pool of `W3_EMIT_THREADS` threads of their own (default 4), each
    // level proving as its own nodes arrive; or (0, the control) one after
    // another on one thread, level 1 waiting for the whole tree's builds.
    let node_pipe = knob("W3_NODE_PIPE").is_none_or(|v| v != 0).then(|| {
        let threads = knob("W3_EMIT_THREADS").unwrap_or(4).max(1);
        rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .thread_name(|i| format!("w3-emit-{i}"))
            .build()
            .expect("the node builder's pool")
    });
    // `W3_DATAFLOW=0|1` (default 1): each node proves as soon as its own
    // children are proved, or (0, the control) once its whole level below is.
    let dataflow = knob("W3_DATAFLOW").is_none_or(|v| v != 0);
    // `W3_STREAM_TOP=0|1` (default 1): when a child is still unproved as the
    // top node's program and artifacts arrive, the top executes each child's
    // share as that child is proved; otherwise, and at 0 (the control), all of
    // it after the last. `W3 TOP EXECUTE` says which the top did.
    let stream_top = knob("W3_STREAM_TOP").is_none_or(|v| v != 0);
    let top_streamed = std::sync::OnceLock::new();
    let t = std::time::Instant::now();
    let tree_unix = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0.0, |d| d.as_secs_f64());
    let (timings, proofs, early_out) = prove_tree_pipelined(
        &plan,
        &proof,
        programs,
        siblings,
        beside,
        early,
        node_pipe.as_ref(),
        dataflow,
        stream_top.then_some(&top_streamed),
    )
    .expect("the tree proves");
    let top = &proofs.last().expect("a top").1;
    let tree = t.elapsed().as_secs_f64();
    let whole = t0.elapsed().as_secs_f64();
    for (lv, level) in timings.iter().enumerate() {
        let programs: Vec<String> = level
            .programs
            .iter()
            .map(|(built, prove)| format!("{built:.2}+{prove:.2}"))
            .collect();
        println!(
            "W3 LEVEL {lv}: {:.2}s wall · per program build+prove {}",
            level.wall,
            programs.join(" · ")
        );
    }
    // `W3_NODE_TIMES=1`: when each program of the tree was built, taken and
    // proved (seconds since the tree started), its prove's split, and for each
    // node level when the level below was done and its nodes were built. Off
    // the clock; the stamps are taken either way.
    if knob("W3_NODE_TIMES") == Some(1) {
        println!("W3 TIMES TREE START: unix {tree_unix:.3} (the card trace's clock)");
        let shape = plan.levels();
        let split = |t: &ProgramTimes| match t.split {
            Some((x, f, p, w)) => format!("exec {x:.2} fill {f:.2} prove {p:.2} wait {w:.2}"),
            None => "no split".to_string(),
        };
        for (lv, level) in timings.iter().enumerate() {
            let mut at = 0usize;
            let rows: Vec<String> = level
                .times
                .iter()
                .enumerate()
                .map(|(j, t)| {
                    let landed = match lv {
                        0 => String::new(),
                        _ => {
                            let a = shape[lv - 1].arities[j];
                            let kids: Vec<String> = timings[lv - 1].times[at..at + a]
                                .iter()
                                .map(|c| format!("{:.2}", c.end))
                                .collect();
                            at += a;
                            format!(" kids done [{}]", kids.join(" "))
                        }
                    };
                    format!(
                        "{j}: built@{:.2}{landed} took@{:.2} done@{:.2} ({})",
                        t.built_at,
                        t.start,
                        t.end,
                        split(t)
                    )
                })
                .collect();
            println!("W3 TIMES L{lv}: {}", rows.join(" · "));
            if lv > 0 {
                let last_kid = timings[lv - 1]
                    .times
                    .iter()
                    .map(|t| t.end)
                    .fold(0.0, f64::max);
                let built = level.times.iter().map(|t| t.built_at).fold(0.0, f64::max);
                let first = level
                    .times
                    .iter()
                    .map(|t| t.start)
                    .fold(f64::INFINITY, f64::min);
                println!(
                    "W3 TIMES START L{lv}: level {} done@{last_kid:.2} · its nodes built by {built:.2} · first node took@{first:.2} ({:+.2} after the later)",
                    lv - 1,
                    first - last_kid.max(built)
                );
            }
        }
    }
    println!(
        "W3 STREAM TOP: {}",
        if stream_top {
            "on (while a child is unproved as its program arrives, the top executes each child's share as that child is proved)"
        } else {
            "off (the top executes after its last child)"
        }
    );
    println!(
        "W3 TOP EXECUTE: {}",
        match (stream_top, top_streamed.get()) {
            (false, _) => "whole (streaming off)",
            (true, Some(true)) =>
                "streamed (a child was still unproved when the top's program and artifacts arrived)",
            (true, Some(false)) =>
                "whole (every child was proved before the top's program and artifacts arrived)",
            (true, None) => "unknown (the top never chose)",
        }
    );
    println!(
        "W3 DATAFLOW: {}",
        if dataflow {
            "on (each node proves as soon as its own children are proved)"
        } else {
            "off (each node waits for its whole level below)"
        }
    );
    println!(
        "W3 NODE PIPE: {}",
        match &node_pipe {
            Some(pool) => format!(
                "on ({} builder threads; each level proves as its own nodes arrive)",
                pool.current_num_threads()
            ),
            None => "off (one builder thread; level 1 waits for the whole tree)".to_string(),
        }
    );
    println!(
        "W3 RECURSION: {:.2}s after the base (tree {tree:.2}s) · whole block {whole:.2}s · whole excl. harness readouts {:.2}s (readouts {readouts:.2}s) · leaves execute beside their artifacts {beside}",
        whole - base,
        whole - readouts
    );
    // Off the clock: the early leaf's arena, built from the groups as phase B
    // handed them over, is the one the finished proof gives.
    match &early_out {
        Some((k, arena, complete_at, started, secs)) => {
            let same = block_leaf_arena(&plan, &proof, *k).expect("the leaf's arena") == *arena;
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
    let ok = verify_block_whir(&proof, &elf, &opts, &format).expect("the verifier runs");
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
