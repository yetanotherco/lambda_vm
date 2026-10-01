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
use multilinear::whir_chain::StackVars;
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
    BLOCK_FAN_IN, BlockPartition, LeafChecks, WhirBlockPlan, artifacts_of, block_leaf_arena,
    id_words, leaf_program_with, out_halves, partition_groups, verify_block_tree,
    verify_block_tree_under,
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
    let elf = asm_elf_bytes(name);
    let opts = ProofOptions::default_test_options();
    let proof = prove_block_whir(
        &elf,
        &[],
        &opts,
        format,
        &BlockOptions {
            max_rows: MaxRowsConfig::small(),
            keccak_rnd_rows_log2: 3,
            drop_levels: 3,
            window_log2: None,
            stream_keccak_rnd: false,
            stream_memw_lt: false,
        },
    )
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

/// The leaf's bus share divides `p` by `q`: a zero `q` has no satisfying
/// assignment, which is the host's `contribution() = None` refusal.
#[test]
fn a_zero_denominator_has_no_satisfying_assignment() {
    let run = |q: u64| {
        let mut b = LfmBuilder::new().with_wrap_hash(super::edsl::WrapHash::production());
        let a = b.declare_arena(2);
        let p = b.hint_word(a, 0).as_ext();
        let q_wire = b.hint_word(a, 1).as_ext();
        let share = b.ediv(p, q_wire);
        b.public(share.as_cell());
        let program = compile(b.finish());
        let words = vec![
            super::word::ext_word(&FEE::from(7u64)),
            super::word::ext_word(&FEE::from(q)),
        ];
        execute(&program, &[words], &crate::hash_pin::BLOCK_HASHER).map(|_| ())
    };
    assert!(run(3).is_ok());
    assert!(run(0).is_err());
}

/// The production format with three polynomials a group: the dense fixture's
/// pages share groups with tables of their height.
fn dense_format() -> BlockFormat {
    BlockFormat {
        zf: ZfFormat::DEFAULT,
        group_polys: block_whir::BLOCK_GROUP_POLYS,
        max_groups: block_whir::BLOCK_MAX_GROUPS,
        prepared: true,
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
            drop_levels: 3,
            window_log2: None,
            stream_keccak_rnd: false,
            stream_memw_lt: false,
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
}

/// One level's readout: its wall, and per program (artifacts, prove) seconds.
struct LevelTiming {
    wall: f64,
    programs: Vec<(f64, f64)>,
}

/// The tree proved the way a prover would run it, with the leaves' programs
/// emitted beforehand (`leaves`, while phase B ran):
/// 1. the leaves' artifacts, in parallel;
/// 2. the nodes' programs and artifacts — functions of the leaves' artifacts,
///    not of any proof — on a thread of their own, beside
/// 3. the leaves proved `siblings` at a time, each harvested without its
///    verify;
/// 4. each level above proved over the harvested children.
///
/// Returns each level's timing and every proof with its artifacts, level by
/// level, the top last — for the harness to verify off the clock.
#[allow(clippy::type_complexity)]
fn prove_tree_pipelined(
    plan: &WhirBlockPlan,
    proof: &BlockWhirProof,
    leaves: Vec<LfmProgram>,
    siblings: usize,
) -> Result<
    (
        Vec<LevelTiming>,
        Vec<(super::registry::LfmArtifacts, LfmProof)>,
    ),
    String,
> {
    use super::per_table_aggregator_tests::{harvest_child, in_index_order};
    let wrap = aggregation_wrap_options();
    let words = plan.child_layout().total();
    let t_level = std::time::Instant::now();

    // 1. the leaves' artifacts.
    let leaf_built: Vec<(super::registry::LfmArtifacts, DerivedChild, f64)> = {
        use rayon::prelude::*;
        leaves
            .par_iter()
            .map(|program| -> Result<_, String> {
                let t = std::time::Instant::now();
                let artifacts = artifacts_of(program, &wrap);
                let derived = DerivedChild::from_artifacts(&artifacts, &wrap, words)?;
                Ok((artifacts, derived, t.elapsed().as_secs_f64()))
            })
            .collect::<Result<_, String>>()?
    };
    let shape = plan.levels();
    let mut timings = Vec::with_capacity(shape.len() + 1);
    let mut proofs = Vec::new();

    let (leaf_proved, nodes) = std::thread::scope(|scope| {
        // 2. the nodes' programs, from the leaves' derived shapes.
        let nodes = scope.spawn(|| -> Result<Vec<Vec<TreeNode>>, String> {
            let mut owned: Vec<Vec<TreeNode>> = Vec::with_capacity(shape.len());
            for (lv, arities) in shape.iter().enumerate() {
                let top = lv + 1 == shape.len();
                let level = {
                    let below: Vec<&DerivedChild> = match lv {
                        0 => leaf_built.iter().map(|(_, d, _)| d).collect(),
                        _ => owned[lv - 1].iter().map(|n| &n.derived).collect(),
                    };
                    let mut at = 0usize;
                    let mut level = Vec::with_capacity(arities.arities.len());
                    for &a in &arities.arities {
                        let t = std::time::Instant::now();
                        let program = plan.node_program(&below[at..at + a], top)?;
                        let artifacts = artifacts_of(&program, &wrap);
                        let derived = DerivedChild::from_artifacts(&artifacts, &wrap, words)?;
                        level.push(TreeNode {
                            program,
                            artifacts,
                            derived,
                            built: t.elapsed().as_secs_f64(),
                        });
                        at += a;
                    }
                    level
                };
                owned.push(level);
            }
            Ok(owned)
        });
        // 3. the leaves, `siblings` at a time.
        let proved = in_index_order(leaves.len(), siblings, |k| -> Result<_, String> {
            let arenas = block_leaf_arena(plan, proof, k)?;
            let t = std::time::Instant::now();
            let lfm = lfm_prove(&leaves[k], &leaf_built[k].0, &arenas, &wrap)
                .map_err(|e| format!("leaf {k}: {e:?}"))?;
            let prove = t.elapsed().as_secs_f64();
            let child = harvest_child(leaf_built[k].0.clone(), wrap.clone(), &lfm);
            Ok((lfm, child, prove))
        });
        (proved, nodes.join())
    });
    let leaf_proved: Vec<(LfmProof, RealChild, f64)> =
        leaf_proved.into_iter().collect::<Result<_, String>>()?;
    let nodes = nodes.map_err(|_| "the node builder panicked".to_string())??;
    timings.push(LevelTiming {
        wall: t_level.elapsed().as_secs_f64(),
        programs: leaf_built
            .iter()
            .zip(&leaf_proved)
            .map(|((_, _, built), (_, _, prove))| (*built, *prove))
            .collect(),
    });
    let mut children: Vec<RealChild> = Vec::with_capacity(leaf_proved.len());
    for ((lfm, child, _), (artifacts, _, _)) in leaf_proved.into_iter().zip(leaf_built) {
        proofs.push((artifacts, lfm));
        children.push(child);
    }

    // 4. the levels above.
    for (lv, (arities, level)) in shape.iter().zip(nodes).enumerate() {
        let t_level = std::time::Instant::now();
        let mut at = 0usize;
        let mut starts = Vec::with_capacity(arities.arities.len());
        for &a in &arities.arities {
            starts.push(at..at + a);
            at += a;
        }
        let proved = in_index_order(level.len(), siblings, |j| -> Result<_, String> {
            let arenas: Vec<Vec<LfmWord>> = children[starts[j].clone()]
                .iter()
                .flat_map(child_arena_words)
                .collect();
            let t = std::time::Instant::now();
            let lfm = lfm_prove(&level[j].program, &level[j].artifacts, &arenas, &wrap)
                .map_err(|e| format!("level {} node {j}: {e:?}", lv + 1))?;
            let prove = t.elapsed().as_secs_f64();
            let child = harvest_child(level[j].artifacts.clone(), wrap.clone(), &lfm);
            Ok((lfm, child, prove))
        });
        let proved: Vec<(LfmProof, RealChild, f64)> =
            proved.into_iter().collect::<Result<_, String>>()?;
        timings.push(LevelTiming {
            wall: t_level.elapsed().as_secs_f64(),
            programs: level
                .iter()
                .zip(&proved)
                .map(|(n, (_, _, prove))| (n.built, *prove))
                .collect(),
        });
        children = Vec::with_capacity(proved.len());
        for ((lfm, child, _), node) in proved.into_iter().zip(level) {
            proofs.push((node.artifacts, lfm));
            children.push(child);
        }
    }
    if children.len() != 1 {
        return Err(format!("the tree closes to {} nodes", children.len()));
    }
    Ok((timings, proofs))
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
    let format = BlockFormat::production();
    let mut options = BlockOptions::production();
    // `BLOCK_WHIR_STREAM_KECCAK_RND=1`: KECCAK_RND's chunks streamed (off by
    // default: NO EFFECT on this block, FAST 418).
    options.stream_keccak_rnd =
        std::env::var("BLOCK_WHIR_STREAM_KECCAK_RND").is_ok_and(|v| v.trim() == "1");
    // `BLOCK_WHIR_STREAM_MEMW_LT=1`: the MEMW-derived LT ops streamed per window.
    options.stream_memw_lt =
        std::env::var("BLOCK_WHIR_STREAM_MEMW_LT").is_ok_and(|v| v.trim() == "1");
    let opts = super::proof::block_base_options();
    let wrap = aggregation_wrap_options();
    super::device_permit::arm(siblings);
    println!(
        "W3 CONFIG: leaves {leaves:?} · siblings {siblings} · fan-in {fan_in} · KECCAK_RND streamed {} · MEMW LT streamed {}",
        options.stream_keccak_rnd, options.stream_memw_lt
    );

    let t0 = std::time::Instant::now();
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
    let (elf_ref, opts_ref, format_ref) = (&elf, &opts, &format);
    let (proved, pre) = std::thread::scope(|scope| {
        let pre = scope.spawn(
            move || -> Result<(WhirBlockPlan, Vec<LfmProgram>, f64, f64), String> {
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
                Ok((plan, programs, at, t0.elapsed().as_secs_f64()))
            },
        );
        let proved =
            block_whir::prove_block_whir_observed(&elf, &input, &opts, &format, &options, &observe);
        // A prover that failed before stating must not leave the planner waiting.
        *sender.lock().expect("lock") = None;
        (proved, pre.join())
    });
    let (proof, stamps) = proved.expect("the block proves");
    let base = t0.elapsed().as_secs_f64();
    let (plan, programs, stated_at, ready_at) = pre
        .expect("the planner did not panic")
        .expect("the plan and the leaves derive");
    print!("{}", stamps.report());
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

    let t = std::time::Instant::now();
    let (timings, proofs) =
        prove_tree_pipelined(&plan, &proof, programs, siblings).expect("the tree proves");
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
    println!(
        "W3 RECURSION: {:.2}s after the base (tree {tree:.2}s) · whole block {whole:.2}s",
        whole - base
    );

    // Off the clock: the harness's checks.
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
    let verdict = if leaves.is_none() && fan_in == BLOCK_FAN_IN {
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
