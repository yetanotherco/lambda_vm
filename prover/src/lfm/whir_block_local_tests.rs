//! The block's in-guest verifier over a base under local lookups
//! ([`BlockFormat::local_lookups`]): each group with a KECCAK_RND table argues
//! its byte lookups under its own challenges, drawn from its commit-time
//! transcript, and the leaf that holds the group checks its local ladders,
//! its local balance, its local pairs in the constraint phase and its
//! LOCAL_BYTES table's fixed columns.
//!
//! - The leaves execute over an honest base under the BLAKE3 base and the
//!   Poseidon1 base, and their shares close the block's bus.
//! - Each group's `(z_g, α_g)` is the host verifier's
//!   (`multilinear_block::local_challenges`); a transcript that skips its
//!   roots, its answers or its digests, or draws before them, is not.
//! - Every mutant of the leaf's local checks refuses the honest base, but
//!   the two that drop a check: those accept what only that check refuses —
//!   a false local lookup (the local balance) and a LOCAL_BYTES table that is
//!   not the verifier's (its fixed columns).
//! - A base drawing its local challenges before its roots is refused, and
//!   the leaf that drops the roots accepts it: the leaf binds what the host
//!   binds.
//! - Tampered and faulted local argues, shapes off the plan's, and each
//!   format's base under the other's plan are refused.
//!
//! Box tier (each proves a small block): run with `--ignored
//! --test-threads=1`. The faults are thread-local; the prover's
//! (`Deviations`) are the prove's own.

use crate::block_whir::{
    self, AnyBlockWhirProof, BlockFormat, BlockWhirProof, Deviations, LocalArgues, LocalBytesFault,
    LocalPart,
};
use crate::hash_pin::{BlockFamily, ProofFamily};
use crate::tables::keccak_rnd::KeccakRndForm;
use crate::tables::types::{FE, FEE, GoldilocksExtension};
use crate::test_utils::asm_elf_bytes;
use crate::tests::block_whir_p1_tests::{many_groups, options};
use multilinear::whir_chain::ArgueFormat;
use stark::multilinear_table::batched::{LocalBinding, ProverFaults};
use stark::proof::options::{BaseFormat, ProofOptions};

use super::compiler::LfmProgram;
use super::executor::execute;
use super::whir_block::local_faults::{self, Fault};
use super::whir_block::{WhirBlockPlan, block_leaf_arena, local_challenges_program};
use super::word::{LfmWord, word_as_ext};

fn opts() -> ProofOptions {
    ProofOptions::default_test_options()
}

/// `base` under KECCAK_RND's `form`, batched (local lookups are argued
/// batched alone).
fn format(base: BaseFormat, form: KeccakRndForm) -> BlockFormat {
    BlockFormat {
        argue: ArgueFormat::BATCHED,
        keccak_rnd: form,
        ..many_groups(base)
    }
}

fn b3_local() -> BlockFormat {
    format(BaseFormat::B3_WHIR, KeccakRndForm::LeanLocal)
}

fn block_options() -> block_whir::BlockOptions {
    let mut o = options(0);
    o.window_log2 = Some(3);
    o
}

fn elf() -> Vec<u8> {
    asm_elf_bytes("test_keccak")
}

/// Under `format`'s base, with the base-only permit a BLAKE3 base needs.
fn permitted<R>(format: &BlockFormat, f: impl FnOnce() -> R) -> R {
    if format.zf.base == BaseFormat::B3_WHIR {
        crate::base_only::with_permit(f)
    } else {
        f()
    }
}

/// The block proved under local lookups with `deviations`.
fn prove_local(
    format: &BlockFormat,
    deviations: &Deviations,
) -> Result<BlockWhirProof<LocalArgues>, crate::Error> {
    permitted(format, || {
        block_whir::prove_block_whir_local_with(
            &elf(),
            &[],
            &opts(),
            format,
            &block_options(),
            deviations,
        )
        .map(|(proof, _)| proof)
    })
}

/// The honest block under `format`, proved once a format.
fn honest(format: &BlockFormat) -> BlockWhirProof<LocalArgues> {
    static B3: std::sync::OnceLock<BlockWhirProof<LocalArgues>> = std::sync::OnceLock::new();
    static P1: std::sync::OnceLock<BlockWhirProof<LocalArgues>> = std::sync::OnceLock::new();
    let cell = if format.zf.base == BaseFormat::B3_WHIR {
        &B3
    } else {
        &P1
    };
    cell.get_or_init(|| {
        let t = std::time::Instant::now();
        let proof = prove_local(format, &Deviations::default()).expect("the local base proves");
        eprintln!(
            "LOCAL REC BASE {:?}: proved in {:.1}s",
            format.zf.base.hash,
            t.elapsed().as_secs_f64()
        );
        proof
    })
    .clone()
}

/// The host verifier's reading of `proof` under `format`: `Ok` it accepts,
/// else why not (a refusal before the proof is read, or the check that
/// refused it).
fn host<L: LocalPart>(proof: &BlockWhirProof<L>, format: &BlockFormat) -> Result<(), String> {
    match permitted(format, || {
        block_whir::verify_block_whir_traced(
            proof,
            &elf(),
            &opts(),
            format,
            false,
            stark::multilinear_table::batched::VerifierChecks::ALL,
        )
    }) {
        Ok(Ok(())) => Ok(()),
        Ok(Err(refusal)) => Err(format!("{refusal:?}")),
        Err(e) => Err(format!("before the proof: {e:?}")),
    }
}

/// The recursion's plan over `statement` under `format`: one leaf (the
/// closed program's) under BLAKE3, one group a leaf under Poseidon1 (the
/// rule's heights hold a small format's KECCAK_W group in no leaf).
fn plan_under(
    format: &BlockFormat,
    statement: block_whir::BlockStatement<'_>,
) -> Result<WhirBlockPlan, String> {
    if format.zf.base == BaseFormat::B3_WHIR {
        WhirBlockPlan::derive_blake3(&elf(), &opts(), format, statement, 1, None)
    } else {
        WhirBlockPlan::derive(
            &elf(),
            &opts(),
            format,
            statement,
            Some(statement.groups.len()),
        )
    }
}

/// The groups that argue local lookups: those carrying a local argue.
fn local_groups(proof: &BlockWhirProof<LocalArgues>) -> Vec<usize> {
    (0..proof.groups.len())
        .filter(|&g| !proof.local[g].is_empty())
        .collect()
}

/// The leaf that holds group `g`.
fn leaf_of(plan: &WhirBlockPlan, g: usize) -> usize {
    (0..plan.partition().num_leaves())
        .find(|&k| plan.partition().leaf(k).contains(&g))
        .expect("a leaf holds every group")
}

/// Leaf `k` emitted under `fault` (the leaf's own mutant, thread-local).
fn leaf_under(plan: &WhirBlockPlan, k: usize, fault: Fault) -> LfmProgram {
    local_faults::with(fault, || plan.leaf_program(k)).expect("the leaf emits")
}

/// `program` executed over leaf `k`'s arena from `proof`: its published
/// words, or why not.
fn execute_leaf<L: LocalPart>(
    program: &LfmProgram,
    plan: &WhirBlockPlan,
    proof: &BlockWhirProof<L>,
    k: usize,
) -> Result<Vec<LfmWord>, String> {
    let arenas = block_leaf_arena(plan, proof, k)?;
    assert_eq!(
        program.arena_schema.lens,
        vec![arenas[0].len() as u32],
        "the leaf hints exactly the words its arena holds"
    );
    execute(
        program,
        &arenas,
        &program.hasher(<BlockFamily as ProofFamily>::SOCKET),
    )
    .map(|e| e.public_words.iter().map(|(_, w)| *w).collect())
    .map_err(|e| format!("{e:?}"))
}

/// Leaf `k` (real, no fault) over `proof`.
fn run_leaf<L: LocalPart>(
    plan: &WhirBlockPlan,
    proof: &BlockWhirProof<L>,
    k: usize,
) -> Result<Vec<LfmWord>, String> {
    execute_leaf(&leaf_under(plan, k, Fault::None), plan, proof, k)
}

/// ★ THE LEAF GATE under local lookups, under the BLAKE3 base (one leaf) and
/// the Poseidon1 base (one group a leaf): the host verifier accepts the
/// base; every leaf executes over it, and their shares cancel (the block's
/// bus closes across them, which only leaves that verified every group
/// against the host's challenges give); the one-leaf plan's closed program
/// (level 1) executes too. Every local group's leaf hints its local argue
/// first; the cost model and the emitted rows of each group are printed.
#[test]
#[ignore = "proves a small block under local lookups; box tier"]
fn the_leaves_of_a_local_base_execute_and_close_the_bus() {
    for format in [
        b3_local(),
        format(BaseFormat::P1_WHIR, KeccakRndForm::LeanLocal),
    ] {
        let label = format!("{:?}", format.zf.base.hash);
        let proof = honest(&format);
        assert_eq!(host(&proof, &format), Ok(()));
        let locals = local_groups(&proof);
        assert!(
            !locals.is_empty(),
            "{label}: the block argues local lookups"
        );
        let plan = plan_under(&format, proof.statement()).expect("the plan derives");
        let (loads, front) = plan.chip_loads();
        println!("LOCAL REC FRONT {label}: {front:?}");
        for (g, load) in loads.iter().enumerate() {
            println!(
                "LOCAL REC GROUP {label} {g}: local {} · tables {} · cost {} (argue {}, opening {}, prepared {}) · rows {:?} · local trees {} · local ladders {}",
                locals.contains(&g),
                proof.groups[g].len(),
                plan.costs()[g],
                plan.cost_parts()[g][0],
                plan.cost_parts()[g][1],
                plan.cost_parts()[g][2],
                load,
                proof.local[g].bus_outputs.len(),
                proof.local[g].gkr.len(),
            );
        }
        let layout = plan.child_layout();
        let mut sum = FEE::zero();
        for k in 0..plan.partition().num_leaves() {
            let words =
                run_leaf(&plan, &proof, k).unwrap_or_else(|e| panic!("{label} leaf {k}: {e}"));
            sum += word_as_ext(&words[layout.sum()]).expect("an extension share");
        }
        assert_eq!(sum, FEE::zero(), "{label}: the leaves' shares cancel");
        if plan.partition().num_leaves() == 1 {
            let program = plan
                .closed_block_program()
                .expect("the closed program emits");
            execute_leaf(&program, &plan, &proof, 0)
                .unwrap_or_else(|e| panic!("{label}: the closed program: {e}"));
        }
    }
}

/// ★ Each local group's `(z_g, α_g)`, drawn in the leaf from the front's root
/// and answer cells and the plan's digests, is the host verifier's
/// (`multilinear_block::local_challenges` over the proof's roots and answers
/// and the verifier's own digests, under the base's transcript); and each
/// mutant of the local transcript's order — the roots skipped, the answers
/// skipped, the label and digests skipped, the draws made first — moves them
/// off the host's. The challenges come after every committed word and every
/// table's definition.
#[test]
#[ignore = "proves a small block under local lookups; box tier"]
fn each_groups_local_challenges_are_the_host_verifiers() {
    use stark::multilinear_block::local_challenges;
    let format = b3_local();
    let proof = honest(&format);
    let plan = plan_under(&format, proof.statement()).expect("the plan derives");
    let frame =
        block_whir::block_frame(proof.statement(), &elf(), &opts(), &format).expect("the frame");
    let refs = frame.airs.air_refs();
    type T = <multilinear::whir_hash::Blake3Whir as multilinear::whir_hash::WhirHash>::Sponge;
    let digests =
        block_whir::local_digests::<T>(&proof.groups, &frame.shapes, |i| refs.get(i).copied())
            .expect("the verifier's digests");
    let arena: Vec<Vec<LfmWord>> = super::whir_block::leaf_arena_under(
        &proof.proof.roots,
        &proof.commit_ood,
        Vec::new(),
        plan.wrap_hash(),
    );
    let draws = |g: usize, fault: Fault| -> Vec<FEE> {
        let program = local_faults::with(fault, || local_challenges_program(&plan, g))
            .expect("the program emits");
        execute(
            &program,
            &arena,
            &program.hasher(<BlockFamily as ProofFamily>::SOCKET),
        )
        .expect("the local challenges execute")
        .public_words
        .iter()
        .map(|(_, w)| word_as_ext(w).expect("an extension draw"))
        .collect()
    };
    let config = &frame.config;
    for g in local_groups(&proof) {
        let layout = &frame.stack_layouts[g];
        let root_at: usize = frame.stack_layouts[..g]
            .iter()
            .map(|l| multilinear::stacked_eval::round0_trees(l.num_polys(), config).len())
            .sum();
        let trees = multilinear::stacked_eval::round0_trees(layout.num_polys(), config).len();
        let answer_at: usize = frame.stack_layouts[..g].iter().map(|l| l.num_polys()).sum();
        let table_at: usize = proof.groups[..g].iter().map(Vec::len).sum();
        let (z, alpha, _) = local_challenges::<GoldilocksExtension, T>(
            g,
            layout,
            &proof.proof.roots[root_at..root_at + trees],
            &proof.commit_ood[answer_at..answer_at + layout.num_polys()],
            &digests[table_at..table_at + proof.groups[g].len()],
            config,
            LocalBinding::ALL,
        )
        .expect("the host's local challenges");
        assert_eq!(draws(g, Fault::None), vec![z, alpha], "group {g}");
        for fault in [
            Fault::NoLocalRoots,
            Fault::NoLocalAnswers,
            Fault::NoLocalDigests,
            Fault::LocalDrawsFirst,
        ] {
            assert_ne!(
                draws(g, fault),
                vec![z, alpha],
                "group {g} {fault:?}: the local challenges must move off the host's"
            );
        }
    }
    let plain = (0..proof.groups.len())
        .find(|g| proof.local[*g].is_empty())
        .expect("a group with no local lookup");
    assert!(
        local_challenges_program(&plan, plain).is_err(),
        "a group with no local lookup draws no local challenge"
    );
}

/// ★ Every mutant of the leaf's local checks has no execution over the honest
/// base — the local transcript's order (roots, answers, digests skipped; the
/// draws first), the fork without the local state or with it last, today's
/// table weights `λ^{3t}` in a local group, LOCAL_BYTES' closed form with OR
/// in XOR's place — while the leaf itself executes: each is load-bearing
/// against an honest prover's bytes. The two mutants that only DROP a check
/// (the local balance, the fixed columns) execute it, as they must; what
/// they let through is the next tests'.
#[test]
#[ignore = "proves a small block under local lookups; box tier"]
fn every_local_mutant_of_the_leaf_refuses_the_honest_base() {
    let format = b3_local();
    let proof = honest(&format);
    let plan = plan_under(&format, proof.statement()).expect("the plan derives");
    let g = local_groups(&proof)[0];
    let k = leaf_of(&plan, g);
    run_leaf(&plan, &proof, k).expect("the honest leaf executes");
    for fault in [
        Fault::NoLocalRoots,
        Fault::NoLocalAnswers,
        Fault::NoLocalDigests,
        Fault::LocalDrawsFirst,
        Fault::NoLocalState,
        Fault::LocalStateLast,
        Fault::TodaysWeights,
        Fault::LocalBytesXorAsOr,
    ] {
        let refused = execute_leaf(&leaf_under(&plan, k, fault), &plan, &proof, k);
        println!("LOCAL REC MUTANT {fault:?}: {:?}", refused.as_ref().err());
        assert!(
            refused.is_err(),
            "{fault:?}: the honest base must have no execution"
        );
    }
    for fault in [Fault::NoLocalBalance, Fault::NoLocalBytesRoute] {
        execute_leaf(&leaf_under(&plan, k, fault), &plan, &proof, k)
            .unwrap_or_else(|e| panic!("{fault:?} drops a check and must execute: {e}"));
    }
}

/// [`a_false_local_lookup_is_refused_by_the_leafs_local_balance`]'s case
/// under `format`: the forgery proves, the host refuses it for the local
/// balance, the leaf that holds its group refuses it, and that leaf without
/// its local-balance assert executes it.
fn false_local_lookup_case(format: &BlockFormat) {
    let format = *format;
    let forged = prove_local(
        &format,
        &Deviations {
            local_bytes: LocalBytesFault::OneMoreReceived,
            ..Deviations::default()
        },
    )
    .expect("the forgery proves");
    let refusal = host(&forged, &format);
    println!("LOCAL REC FALSE LOOKUP host: {refusal:?}");
    assert!(
        refusal
            .as_ref()
            .is_err_and(|e| e.starts_with("Proof(LocalImbalance")),
        "the host refuses for the local balance: {refusal:?}"
    );
    let plan = plan_under(&format, forged.statement()).expect("the plan derives");
    let g = local_groups(&forged)[0];
    let k = leaf_of(&plan, g);
    let refused = run_leaf(&plan, &forged, k);
    println!("LOCAL REC FALSE LOOKUP leaf: {:?}", refused.as_ref().err());
    assert!(refused.is_err(), "the leaf refuses a false local lookup");
    execute_leaf(
        &leaf_under(&plan, k, Fault::NoLocalBalance),
        &plan,
        &forged,
        k,
    )
    .unwrap_or_else(|e| panic!("with no local balance the forgery executes: {e}"));
}

/// ★ A FALSE LOCAL LOOKUP: a prover whose LOCAL_BYTES receives one lookup no
/// table sent (its `MU_ARE_BYTES` one more at a byte pair), every tree,
/// ladder and opening honest over it. The host verifier refuses it for the
/// group's local balance alone (`LocalImbalance`); the leaf that holds the
/// group has no execution over it; and the leaf with its local-balance
/// assert dropped executes it — the assert in the leaf is the check that
/// refuses a false local lookup.
#[test]
#[ignore = "proves a small block under local lookups; box tier"]
fn a_false_local_lookup_is_refused_by_the_leafs_local_balance() {
    false_local_lookup_case(&b3_local());
}

/// ★ The same false local lookup under the two-layer GKR ladder, the BLAKE3
/// base's default (the levels verifier pins it): the host refuses it for the
/// local balance, the leaf refuses it, and the leaf without its local-balance
/// assert executes it.
#[test]
#[ignore = "proves a small block under local lookups; box tier"]
fn a_false_local_lookup_is_refused_by_the_leafs_local_balance_under_the_two_layer_ladder() {
    let format = BlockFormat {
        argue: ArgueFormat::BATCHED.with_ladder(multilinear::whir_chain::GkrLadder::Double),
        ..b3_local()
    };
    assert_eq!(
        format.argue.ladder(),
        multilinear::whir_chain::GkrLadder::Double
    );
    false_local_lookup_case(&format);
}

/// ★ A LOCAL_BYTES TABLE THAT IS NOT THE VERIFIER'S: a prover whose
/// LOCAL_BYTES has `X ⊕ Y` one more at a row no XOR lookup reads, its
/// lookups still balanced, every argue and opening honest over it. The host
/// verifier refuses it (the fixed columns); the leaf has no execution over
/// it; and the leaf with LOCAL_BYTES' route dropped executes it — the closed
/// form is the check that holds the receivers to the verifier's table.
#[test]
#[ignore = "proves a small block under local lookups; box tier"]
fn a_wrong_local_bytes_column_is_refused_by_the_leafs_closed_form() {
    let format = b3_local();
    let forged = prove_local(
        &format,
        &Deviations {
            local_bytes: LocalBytesFault::WrongUnusedXor,
            ..Deviations::default()
        },
    )
    .expect("the forgery proves");
    let refusal = host(&forged, &format);
    println!("LOCAL REC WRONG COLUMN host: {refusal:?}");
    assert!(refusal.is_err(), "the host refuses a wrong fixed column");
    let plan = plan_under(&format, forged.statement()).expect("the plan derives");
    let g = local_groups(&forged)[0];
    let k = leaf_of(&plan, g);
    let refused = run_leaf(&plan, &forged, k);
    println!("LOCAL REC WRONG COLUMN leaf: {:?}", refused.as_ref().err());
    assert!(refused.is_err(), "the leaf refuses a wrong fixed column");
    execute_leaf(
        &leaf_under(&plan, k, Fault::NoLocalBytesRoute),
        &plan,
        &forged,
        k,
    )
    .unwrap_or_else(|e| panic!("with no LOCAL_BYTES route the forgery executes: {e}"));
}

/// ★ A base whose local challenges were drawn before its roots (the prover's
/// mutant binding: the prefix and `z0`, no roots): the host verifier refuses
/// it, the leaf has no execution over it, and the leaf whose local
/// transcript skips the roots executes it — the leaf binds the roots exactly
/// where the host does. (The forgery this enables, counts fixed after the
/// challenges, is the host's test of the same binding.)
#[test]
#[ignore = "proves a small block under local lookups; box tier"]
fn a_base_drawing_its_local_challenges_before_its_roots_is_refused() {
    let format = b3_local();
    let mutant = prove_local(
        &format,
        &Deviations {
            argue: stark::multilinear_block::ArgueDeviation {
                faults: ProverFaults {
                    local_binding: Some(LocalBinding {
                        roots: false,
                        ..LocalBinding::ALL
                    }),
                    ..ProverFaults::default()
                },
                ..Default::default()
            },
            ..Deviations::default()
        },
    )
    .expect("the mutant base proves");
    assert!(host(&mutant, &format).is_err(), "the host refuses it");
    let plan = plan_under(&format, mutant.statement()).expect("the plan derives");
    let g = local_groups(&mutant)[0];
    let k = leaf_of(&plan, g);
    assert!(run_leaf(&plan, &mutant, k).is_err(), "the leaf refuses it");
    execute_leaf(
        &leaf_under(&plan, k, Fault::NoLocalRoots),
        &plan,
        &mutant,
        k,
    )
    .unwrap_or_else(|e| panic!("the leaf without the roots accepts its twin: {e}"));
}

/// ★ A local argue faulted at the prove (a local claim absorbed off by one in
/// the constraint phase) and the honest base tampered after it (a local
/// tree's output, a local ladder's half, a local group's commit-time answer)
/// are refused by the host and have no execution in the leaf that holds the
/// group. (Two tables' local claims traded is the host's test, over tables of
/// one height: a block's K group pairs KECCAK_RND with LOCAL_BYTES, of
/// another height, and its prover cannot trade them.)
#[test]
#[ignore = "proves a small block under local lookups; box tier"]
fn faulted_and_tampered_local_argues_are_refused() {
    let format = b3_local();
    let honest = honest(&format);
    let g = local_groups(&honest)[0];
    let mut sites: Vec<(String, BlockWhirProof<LocalArgues>)> = Vec::new();
    let faulted = prove_local(
        &format,
        &Deviations {
            argue: stark::multilinear_block::ArgueDeviation {
                group: Some(g),
                faults: ProverFaults {
                    local_tamper_claim: Some(0),
                    ..ProverFaults::default()
                },
                ..Default::default()
            },
            ..Deviations::default()
        },
    )
    .expect("the faulted base proves");
    sites.push(("a local claim absorbed off by one".into(), faulted));
    let mut p = honest.clone();
    p.local[g].bus_outputs[0].0 += FEE::one();
    sites.push(("a local tree's output".into(), p));
    let mut p = honest.clone();
    p.local[g].gkr[0].layers[0].halves[0].p_lo += FEE::one();
    sites.push(("a local ladder's half".into(), p));
    let mut p = honest.clone();
    let answer_at: usize = {
        let frame = block_whir::block_frame(honest.statement(), &elf(), &opts(), &format)
            .expect("the frame");
        frame.stack_layouts[..g].iter().map(|l| l.num_polys()).sum()
    };
    p.commit_ood[answer_at] += FEE::one();
    sites.push(("the local group's first answer".into(), p));
    for (what, forged) in &sites {
        assert!(host(forged, &format).is_err(), "{what}: the host refuses");
        let plan = plan_under(&format, forged.statement()).expect("the plan derives");
        let k = leaf_of(&plan, g);
        let refused = run_leaf(&plan, forged, k);
        println!("LOCAL REC TAMPER {what}: {:?}", refused.as_ref().err());
        assert!(refused.is_err(), "{what}: the leaf has no execution");
    }
}

/// ★ The local argue's shape is the plan's, by name: a local argue a tree
/// short or a ladder short, a group with no local lookup carrying one, and a
/// local group carrying an empty one are refused when the arena is read,
/// before any execution.
#[test]
#[ignore = "proves a small block under local lookups; box tier"]
fn the_local_argues_shape_is_the_plans() {
    let format = b3_local();
    let honest = honest(&format);
    let plan = plan_under(&format, honest.statement()).expect("the plan derives");
    let g = local_groups(&honest)[0];
    let plain = (0..honest.groups.len())
        .find(|h| honest.local[*h].is_empty())
        .expect("a group with no local lookup");
    let k = leaf_of(&plan, g);
    let mut edits: Vec<(&str, BlockWhirProof<LocalArgues>)> = Vec::new();
    let mut p = honest.clone();
    p.local[g].bus_outputs.pop();
    edits.push(("a local tree short", p));
    let mut p = honest.clone();
    p.local[g].gkr.pop();
    edits.push(("a local ladder short", p));
    let mut p = honest.clone();
    p.local[plain] = honest.local[g].clone();
    edits.push(("a group with no local lookup carrying a local argue", p));
    let mut p = honest.clone();
    p.local[g] = Default::default();
    edits.push(("a local group carrying none", p));
    let mut p = honest.clone();
    p.local.pop();
    edits.push(("a local argue short", p));
    for (what, edited) in &edits {
        let refused = (0..plan.partition().num_leaves())
            .map(|k| block_leaf_arena(&plan, edited, k))
            .find_map(Result::err);
        println!("LOCAL REC SHAPE {what}: {refused:?}");
        assert!(refused.is_some(), "{what}: refused by the arena");
        assert!(host(edited, &format).is_err(), "{what}: the host refuses");
    }
    run_leaf(&plan, &honest, k).expect("the honest leaf executes");
}

/// ★ Format confusion, both ways: a plan under local lookups refuses today's
/// proof (and the local proof stripped of its local argues); today's plan
/// refuses a statement under local lookups (its LOCAL_BYTES tables are not
/// in today's frame) and today's proof carrying local argues; and either
/// kind held as one type ([`AnyBlockWhirProof`]) is read by its own plan
/// only. The leaf's format is the deriver's, never the proof's.
#[test]
#[ignore = "proves two small blocks; box tier"]
fn each_formats_plan_refuses_the_others_base() {
    let local = b3_local();
    let lean = format(BaseFormat::B3_WHIR, KeccakRndForm::Lean);
    let local_proof = honest(&local);
    let todays: BlockWhirProof = permitted(&lean, || {
        block_whir::prove_block_whir_with(
            &elf(),
            &[],
            &opts(),
            &lean,
            &block_options(),
            &Deviations::default(),
        )
        .expect("today's base proves")
        .0
    });
    let local_plan = plan_under(&local, local_proof.statement()).expect("the local plan");
    let lean_plan = plan_under(&lean, todays.statement()).expect("today's plan");
    run_leaf(&lean_plan, &todays, 0).expect("today's leaf executes today's base");
    // Each refused by the frame, by its own message.
    for (what, refused) in [
        (
            "today's plan over a local statement",
            plan_under(&lean, local_proof.statement()).err(),
        ),
        (
            "a local plan over today's statement",
            plan_under(&local, todays.statement()).err(),
        ),
    ] {
        println!("LOCAL REC FORMAT {what}: {refused:?}");
        assert!(
            refused
                .as_ref()
                .is_some_and(|e| e.starts_with("statement:")),
            "{what}: {refused:?}"
        );
    }
    assert!(
        block_leaf_arena(&local_plan, &local_proof.clone().with_local(()), 0).is_err(),
        "a local plan over a local proof stripped of its local argues"
    );
    let padded = todays
        .clone()
        .with_local(vec![Default::default(); todays.groups.len()]);
    assert!(
        block_leaf_arena(&lean_plan, &padded, 0).is_err(),
        "today's plan over today's proof carrying local argues"
    );
    let any_local: AnyBlockWhirProof = local_proof.clone().with_local(Some(local_proof.local));
    let any_todays: AnyBlockWhirProof = todays.clone().with_local(None);
    run_leaf(&local_plan, &any_local, 0).expect("either kind: the local base");
    run_leaf(&lean_plan, &any_todays, 0).expect("either kind: today's base");
    assert!(block_leaf_arena(&local_plan, &any_todays, 0).is_err());
    assert!(block_leaf_arena(&lean_plan, &any_local, 0).is_err());
}

/// ★ LOCAL_BYTES' closed form ([`emit_local_bytes_preprocessed`]) is its four
/// preprocessed columns' multilinear extensions at a point: executed at a few
/// points, against `Mle::evaluate_in` over `tables::local_bytes`' columns.
/// Not blind: the mutant with OR in XOR's place differs.
///
/// [`emit_local_bytes_preprocessed`]: super::preprocessed::emit_local_bytes_preprocessed
#[test]
fn the_local_bytes_closed_form_is_its_columns_extension() {
    use crate::tables::local_bytes::{LOG_ROWS, NUM_PRECOMPUTED_COLS, preprocessed_columns};
    let columns = preprocessed_columns();
    let point = |seed: u64| -> Vec<FEE> {
        (0..LOG_ROWS as u64)
            .map(|i| {
                let x = seed
                    .wrapping_mul(0x9e37_79b9_7f4a_7c15)
                    .wrapping_add(i * 7919 + 3);
                FEE::new([FE::from(x), FE::from(x >> 7), FE::from(i + 1)])
            })
            .collect()
    };
    let run = |at: &[FEE], fault: Fault| -> Vec<FEE> {
        let mut b = super::builder::LfmBuilder::new();
        let wires: Vec<super::builder::Ext> = at.iter().map(|v| b.ext_const(v)).collect();
        let values = local_faults::with(fault, || {
            super::preprocessed::emit_local_bytes_preprocessed(&mut b, &wires)
        });
        for v in values {
            b.public(v.as_cell());
        }
        let program = super::compiler::compile(b.finish());
        execute(
            &program,
            &[],
            &program.hasher(<BlockFamily as ProofFamily>::SOCKET),
        )
        .expect("the closed form executes")
        .public_words
        .iter()
        .map(|(_, w)| word_as_ext(w).expect("an extension value"))
        .collect()
    };
    for seed in [1u64, 2, 0xdead_beef] {
        let at = point(seed);
        let host: Vec<FEE> = (0..NUM_PRECOMPUTED_COLS)
            .map(|c| {
                let mle = multilinear::mle::Mle::new(columns[c].clone()).expect("a column");
                mle.evaluate_in(&at).expect("evaluates")
            })
            .collect();
        assert_eq!(run(&at, Fault::None), host, "seed {seed}");
        assert_ne!(
            run(&at, Fault::LocalBytesXorAsOr),
            host,
            "seed {seed}: the mutant"
        );
    }
}

/// ★ THE STATIC RECEIVER CHECK at derive time: a LOCAL_BYTES whose
/// `LocalAreBytes` receiver reads a committed column (its own `MU_AND`) is
/// refused by the plan's group check — a group's own challenges could argue
/// any false lookup into such a receiver — and LOCAL_BYTES itself passes
/// under its route's four checked columns, and is refused under a route that
/// checks fewer columns than its receivers read.
/// Not blind: with the check dropped, the dynamic receiver passes.
#[test]
fn a_dynamic_local_receiver_is_refused_by_the_plans_static_check() {
    use crate::tables::local_bytes::{self, LOCAL_BUSES, cols};
    use crate::tables::types::BusId;
    use stark::lookup::{BusInteraction, BusValue, Multiplicity, Packing};
    use stark::traits::AIR;
    let opts = opts();
    let direct = |column: usize| BusValue::Packed {
        start_column: column,
        packing: Packing::Direct,
    };
    let fixed = crate::test_utils::create_local_bytes_air(&opts);
    let mut interactions = local_bytes::bus_interactions();
    interactions[0] = BusInteraction::receiver(
        BusId::LocalAreBytes,
        Multiplicity::Column(cols::MU_ARE_BYTES),
        vec![direct(cols::X), direct(cols::MU_AND)],
    );
    let dynamic = crate::test_utils::create_local_bytes_air_with(&opts, interactions);
    // `checked`: the columns the table's route checks (LOCAL_BYTES' four).
    let check_under = |air: &dyn AIR<
        Field = crate::tables::types::GoldilocksField,
        FieldExtension = GoldilocksExtension,
        PublicInputs = (),
    >,
                       checked: usize| {
        let refs = [air];
        let shapes = [(air.trace_layout().0, local_bytes::LOG_ROWS)];
        let owned = super::whir_block::Shapes::build(&refs, &shapes, &[0]).expect("the shapes");
        owned.check_local_receivers(&[checked], &LOCAL_BUSES)
    };
    let check = |air: &dyn AIR<
        Field = crate::tables::types::GoldilocksField,
        FieldExtension = GoldilocksExtension,
        PublicInputs = (),
    >| check_under(air, local_bytes::NUM_PRECOMPUTED_COLS);
    assert_eq!(
        check(&fixed),
        Ok(()),
        "LOCAL_BYTES reads fixed columns only"
    );
    // The predicate is the route's count: under a route that leaves Y
    // unchecked, LOCAL_BYTES itself is refused.
    assert!(
        check_under(&fixed, cols::Y).is_err_and(|e| e.contains("LocalReceiverNotFixed")),
        "a route that leaves Y unchecked"
    );
    let refused = check(&dynamic);
    println!("LOCAL REC DYNAMIC RECEIVER: {refused:?}");
    assert!(
        refused
            .as_ref()
            .is_err_and(|e| e.contains("LocalReceiverNotFixed")),
        "a receiver on a committed column: {refused:?}"
    );
    assert_eq!(
        local_faults::with(Fault::NoReceiverCheck, || check(&dynamic)),
        Ok(()),
        "the mutant check passes it"
    );
}

/// ★ The local transcript reads the FRONT's root and answer cells — the
/// cells the main transcript absorbed and the opening checks — never a copy
/// of its own: the leaf hints exactly the words the base's arena holds (a
/// hinted copy would lengthen it), and the mutant that reads hinted copies
/// needs one word more per root word and per answer of each local group.
/// Fed the honest copies, that mutant executes the honest base — nothing in
/// the program ties its copies to the committed words, which is why a copy
/// would let a prover choose its local challenges' inputs apart from what it
/// committed — and fed one changed copy it does not (its challenges move).
#[test]
#[ignore = "proves a small block under local lookups; box tier"]
fn the_local_transcript_reads_the_fronts_cells_not_hints() {
    let format = b3_local();
    let proof = honest(&format);
    let plan = plan_under(&format, proof.statement()).expect("the plan derives");
    let g = local_groups(&proof)[0];
    let k = leaf_of(&plan, g);
    let real = leaf_under(&plan, k, Fault::None);
    let arena = block_leaf_arena(&plan, &proof, k).expect("the arena");
    assert_eq!(real.arena_schema.lens, vec![arena[0].len() as u32]);
    let mutant = leaf_under(&plan, k, Fault::HintedLocalCells);
    let frame =
        block_whir::block_frame(proof.statement(), &elf(), &opts(), &format).expect("the frame");
    let config = &frame.config;
    let trees = |l: &multilinear::stacking::StackedLayout| {
        multilinear::stacked_eval::round0_trees(l.num_polys(), config).len()
    };
    let locals: Vec<usize> = plan
        .partition()
        .leaf(k)
        .iter()
        .copied()
        .filter(|h| !proof.local[*h].is_empty())
        .collect();
    // Each local group's copies, at the start of its words.
    let (_, marks) = plan.leaf_program_marked(k).expect("the marks");
    let copies = |h: usize, bump: bool| -> Vec<LfmWord> {
        let layout = &frame.stack_layouts[h];
        let root_at: usize = frame.stack_layouts[..h].iter().map(trees).sum();
        let answer_at: usize = frame.stack_layouts[..h].iter().map(|l| l.num_polys()).sum();
        let mut words: Vec<LfmWord> = proof.proof.roots[root_at..root_at + trees(layout)]
            .iter()
            .flat_map(|root| super::proof_arena::commitment_words_for(root, plan.wrap_hash()))
            .collect();
        words.extend(
            proof.commit_ood[answer_at..answer_at + layout.num_polys()]
                .iter()
                .map(super::word::ext_word),
        );
        if bump {
            words[0][0] += FE::one();
        }
        words
    };
    let extra: usize = locals.iter().map(|&h| copies(h, false).len()).sum();
    assert_eq!(
        mutant.arena_schema.lens,
        vec![(arena[0].len() + extra) as u32],
        "the mutant hints a copy of each local group's roots and answers"
    );
    let with_copies = |bump: bool| -> Vec<Vec<LfmWord>> {
        let mut words = arena[0][..marks[0] as usize].to_vec();
        for (i, &h) in plan.partition().leaf(k).iter().enumerate() {
            if locals.contains(&h) {
                words.extend(copies(h, bump));
            }
            words.extend_from_slice(&arena[0][marks[i] as usize..marks[i + 1] as usize]);
        }
        vec![words]
    };
    let run = |arenas: &[Vec<LfmWord>]| {
        execute(
            &mutant,
            arenas,
            &mutant.hasher(<BlockFamily as ProofFamily>::SOCKET),
        )
        .map(|_| ())
        .map_err(|e| format!("{e:?}"))
    };
    run(&with_copies(false)).expect("the mutant over the honest copies executes");
    assert!(
        run(&with_copies(true)).is_err(),
        "a changed copy moves the mutant's local challenges"
    );
}

/// ★ The local balance's shares are `p · (1/q)` whatever the leaves' share
/// knob: under the quotient form (`share_inverse` off) the leaf emits the
/// same local balance as under the inverse form, and the mutant whose local
/// shares follow the knob emits one product fewer per local tree there —
/// a quotient share is free at `p = q = 0`. Under the inverse form the
/// mutant is the leaf.
#[test]
#[ignore = "proves a small block under local lookups; box tier"]
fn the_local_balance_divides_p_by_q_whatever_the_share_knob() {
    let format = b3_local();
    let proof = honest(&format);
    let plan = plan_under(&format, proof.statement()).expect("the plan derives");
    let g = local_groups(&proof)[0];
    let k = leaf_of(&plan, g);
    let trees: usize = plan
        .partition()
        .leaf(k)
        .iter()
        .map(|&h| proof.local[h].bus_outputs.len())
        .sum();
    assert!(trees > 0);
    let quotient = plan.with_share_inverse(false);
    let real = leaf_under(&quotient, k, Fault::None);
    let knob = leaf_under(&quotient, k, Fault::LocalShareKnob);
    assert_eq!(
        real.instrs.len(),
        knob.instrs.len() + trees,
        "one product (p · 1/q) a local tree that the knob's quotient form drops"
    );
    let inverse = quotient.with_share_inverse(true);
    assert_eq!(
        leaf_under(&inverse, k, Fault::None).instrs.len(),
        leaf_under(&inverse, k, Fault::LocalShareKnob).instrs.len(),
        "under the inverse form the mutant is the leaf"
    );
}
