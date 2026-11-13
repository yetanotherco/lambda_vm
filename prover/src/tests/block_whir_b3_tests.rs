//! D-B3HYB S3: the block's base proof under standard 7-round BLAKE3
//! ([`BaseFormat::B3_WHIR`], `Blake3Whir`) — an experiment arm admitted only
//! to a base proved with no recursion over it ([`crate::base_only`]).
//!
//! Pinned here: the default is Poseidon1 and dispatches exactly as before; a
//! BLAKE3 base is refused without the base-only permit and by the recursion
//! even with it; with it, a BLAKE3 base proves and verifies on the host, a
//! Poseidon1 verifier refuses it (and the other way round), and its tampers —
//! an opened leaf value, a root, an absorbed out-of-domain answer, a grind
//! nonce — are refused.

use crate::block_whir::{
    B3_BASE_TAG_SUFFIX, BlockFormat, BlockRefusal, BlockWhirProof, Deviations, block_statement_tag,
    checked_base, checked_base_only, prove_block_whir_with, verify_block_whir,
    verify_block_whir_traced,
};
use crate::statement::MULTILINEAR_BLOCK_TAG;
use crate::test_utils::asm_elf_bytes;
use crate::tests::block_whir_p1_tests::{many_groups, options};
use crate::zf_format::ZfFormat;
use math::field::element::FieldElement;
use multilinear::whir_chain::{LeafOrder, RoundOpenings};
use stark::multilinear_table::batched::VerifierChecks;
use stark::proof::options::{BaseFormat, ProofOptions};

type F = crate::tables::types::GoldilocksField;
type E = crate::tables::types::GoldilocksExtension;

fn prove(elf: &[u8], format: &BlockFormat, drop_levels: usize) -> BlockWhirProof {
    prove_block_whir_with(
        elf,
        &[],
        &ProofOptions::default_test_options(),
        format,
        &options(drop_levels),
        &Deviations::default(),
    )
    .expect("prove")
    .0
}

fn verify(proof: &BlockWhirProof, elf: &[u8], format: &BlockFormat) -> bool {
    verify_block_whir(proof, elf, &ProofOptions::default_test_options(), format).unwrap_or(false)
}

/// The refusal a verifier gives `proof`, by cause, or `None` when it accepts.
fn refusal(proof: &BlockWhirProof, elf: &[u8], format: &BlockFormat) -> Option<String> {
    match verify_block_whir_traced(
        proof,
        elf,
        &ProofOptions::default_test_options(),
        format,
        false,
        VerifierChecks::ALL,
    ) {
        Ok(Ok(())) => None,
        Ok(Err(BlockRefusal::Proof(e))) => Some(format!("{e:?}")),
        Ok(Err(other)) => Some(format!("{other:?}")),
        Err(e) => Some(format!("{e:?}")),
    }
}

/// The configuration `base` dispatches the base's sites to, by type name.
fn dispatched(base: BaseFormat) -> Result<&'static str, crate::Error> {
    Ok(crate::with_base_hash!(base, |H| {
        std::any::type_name::<H>()
    }))
}

/// ★ The default is today's: the format's base is Poseidon1, its statement tag
/// is unmoved, its banner names no base, and the base's dispatch binds
/// `P1Whir` with or without the base-only permit.
#[test]
fn the_default_base_is_poseidon1_and_dispatches_as_before() {
    assert_eq!(ZfFormat::DEFAULT.base, BaseFormat::P1_WHIR);
    assert_eq!(crate::hash_pin::BLOCK_BASE, BaseFormat::P1_WHIR);
    assert_eq!(ZfFormat::DEFAULT.whir_leaves, LeafOrder::Coset);
    assert!(!ZfFormat::DEFAULT.banner().contains("base="));
    assert_eq!(
        block_statement_tag(&BaseFormat::P1_WHIR),
        b"LAMBDAVM_MULTILINEAR_BLOCK_STATEMENT_V2/P1W16/C2".to_vec()
    );
    let p1 = std::any::type_name::<multilinear::whir_hash::P1Whir>();
    assert_eq!(dispatched(BaseFormat::P1_WHIR).expect("P1"), p1);
    crate::base_only::with_permit(|| {
        assert_eq!(dispatched(BaseFormat::P1_WHIR).expect("P1"), p1);
    });
    assert!(dispatched(BaseFormat::RPX).is_err(), "RPX is still refused");
    assert!(crate::base_only::refuse_recursion().is_ok());
}

/// ★ A BLAKE3 base is admitted to the base's sites only under the base-only
/// permit, and never to the recursion's (`checked_base`, `with_block_hash!`),
/// permit or not. Its tag is its own.
#[test]
fn a_blake3_base_needs_the_base_only_permit_and_never_reaches_the_recursion() {
    let b3 = BaseFormat::B3_WHIR;
    assert!(checked_base_only(&b3).is_err(), "no permit");
    assert!(dispatched(b3).is_err(), "no permit");
    assert!(checked_base(&b3).is_err());
    crate::base_only::with_permit(|| {
        assert!(checked_base_only(&b3).is_ok());
        assert_eq!(
            dispatched(b3).expect("B3 under the permit"),
            std::any::type_name::<multilinear::whir_hash::Blake3Whir>()
        );
        assert!(checked_base(&b3).is_err(), "the recursion's check");
        let recursion = || -> Result<&'static str, crate::Error> {
            Ok(crate::with_block_hash!(b3, |H| {
                std::any::type_name::<H>()
            }))
        };
        assert!(recursion().is_err(), "the recursion's dispatch");
        // Another hash at a 4-ary cap is not the BLAKE3 base.
        let capped = BaseFormat {
            arity4_cap: stark::proof::options::CapPolicy::Fixed(2),
            ..b3
        };
        assert!(checked_base_only(&capped).is_err());
    });
    let mut want = MULTILINEAR_BLOCK_TAG.to_vec();
    want.extend_from_slice(B3_BASE_TAG_SUFFIX);
    assert_eq!(block_statement_tag(&b3), want);
    assert_ne!(
        block_statement_tag(&b3),
        block_statement_tag(&BaseFormat::P1_WHIR)
    );
}

/// The recursion's entry points refuse a base-only process and a non-family
/// base before anything is proved: the tree's prover and the plan both call
/// the refusal and the family's check (a process latch cannot be set in the
/// test process, so the calls are read from the source).
#[test]
fn the_recursion_entry_points_refuse_a_base_only_process() {
    let tree = include_str!("../lfm/whir_block_tree.rs");
    let body = &tree[tree
        .find("pub fn prove_whir_block_tree(")
        .expect("the tree's prover")..];
    let head = &body[..body
        .find("sink.line(&posture_line());")
        .expect("its first line")];
    assert!(head.contains("crate::base_only::refuse_recursion()?;"));
    assert!(head.contains("block_whir::checked_base(&format.zf.base)"));
    let plan = include_str!("../lfm/whir_block.rs");
    let derive = &plan[plan.find("fn derive_full(").expect("the plan")..];
    let derive = &derive[..derive.find("let wrap_hash").expect("its check")];
    assert!(derive.contains("crate::base_only::refuse_recursion()?;"));
    assert!(derive.contains("crate::block_whir::checked_base(base)"));
}

/// ★★ Under the permit a BLAKE3 base proves and verifies on the host, with
/// groups retired and revived between the phases (drop 3), in the coset leaf
/// order; a Poseidon1 verifier refuses it and a BLAKE3 verifier refuses a
/// Poseidon1 proof; and each tamper is refused: an opened round-0 leaf value
/// (its Merkle path no longer opens it), a committed root, an absorbed
/// commit-time out-of-domain answer, a round's absorbed out-of-domain value,
/// and a query grind's nonce.
#[test]
fn a_blake3_base_proves_verifies_and_refuses_its_tampers() {
    let elf = asm_elf_bytes("all_instructions_64");
    let (b3, p1) = (
        many_groups(BaseFormat::B3_WHIR),
        many_groups(BaseFormat::P1_WHIR),
    );
    assert_eq!(b3.zf.whir_leaves, LeafOrder::Coset);
    let p1_proof = prove(&elf, &p1, 3);
    crate::base_only::with_permit(|| {
        let honest = prove(&elf, &b3, 3);
        assert!(honest.proof.columns.len() > 1, "several groups");
        assert_eq!(
            refusal(&honest, &elf, &b3),
            None,
            "the BLAKE3 base verifies"
        );
        assert!(
            !verify(&honest, &elf, &p1),
            "a BLAKE3 proof under Poseidon1"
        );
        assert!(
            !verify(&p1_proof, &elf, &b3),
            "a Poseidon1 proof under BLAKE3"
        );
        assert_ne!(honest.proof.roots, p1_proof.proof.roots);

        let tampered = |name: &str, edit: &dyn Fn(&mut BlockWhirProof)| {
            let mut proof = honest.clone();
            edit(&mut proof);
            let why = refusal(&proof, &elf, &b3);
            println!("B3 TAMPER {name}: {why:?}");
            assert!(why.is_some(), "{name}: accepted");
        };
        // An opened leaf value of round 0, chain 0 of group 0.
        tampered(
            "round-0 leaf value",
            &|p| match &mut p.proof.columns[0].polys[0].rounds[0].openings {
                RoundOpenings::Base(round) => {
                    round.current[0].values[0] += FieldElement::<F>::one();
                }
                other => panic!("a per-polynomial round 0 opens base blocks: {other:?}"),
            },
        );
        // A committed root: absorbed before every challenge, and every path of
        // its tree is checked against it.
        tampered("root byte", &|p| p.proof.roots[0][0] ^= 1);
        // The commit-time out-of-domain answers, absorbed after the roots.
        tampered("commit-time OOD answer", &|p| {
            p.commit_ood[0] += FieldElement::<E>::one();
        });
        // Round 0's out-of-domain value of its successor, absorbed in the
        // chain's transcript.
        tampered("round OOD value", &|p| {
            let round = &mut p.proof.columns[0].polys[0].rounds[0];
            let value = round.ood_value.as_mut().expect("round 0 has a successor");
            *value += FieldElement::<E>::one();
        });
        // A query grind's nonce. The block's verifier reports a chain's
        // refusal as its column's opening; that the grind itself is what
        // refuses is pinned where the cause is visible
        // (`multilinear::whir_coset_tests::a_blake3_grind_nonce_moved_is_refused_by_the_grind`).
        tampered("query nonce", &|p| {
            p.proof.columns[0].polys[0].rounds[0].nonces.query ^= 1;
        });
    });
    // Without the permit the BLAKE3 base is not even read.
    let made = crate::base_only::with_permit(|| prove(&elf, &b3, 0));
    assert!(
        verify_block_whir(&made, &elf, &ProofOptions::default_test_options(), &b3).is_err(),
        "refused, not verified"
    );
}

/// The BLAKE3 base in the natural leaf order proves and verifies too, and is
/// refused under the coset order: the leaf order is a verifier constant under
/// either hash.
#[test]
fn a_blake3_base_in_the_natural_order_is_its_own_format() {
    let elf = asm_elf_bytes("sub");
    let coset = many_groups(BaseFormat::B3_WHIR);
    let mut natural = coset;
    natural.zf.whir_leaves = LeafOrder::Natural;
    crate::base_only::with_permit(|| {
        let proof = prove(&elf, &natural, 0);
        assert!(verify(&proof, &elf, &natural));
        assert!(!verify(&proof, &elf, &coset));
    });
}

/// ★★ The BLAKE3 base with grouped chains takes one round-0 tree a run by
/// default (the format's `whir_trees`, as under Poseidon1): under the permit
/// it proves and verifies with groups retired and revived (drop 3), its
/// grouped runs open joint leaves, it carries fewer roots than the same block
/// under one tree a polynomial (`each`, the rollback, which proves and
/// verifies too), and each format's verifier refuses the other's proof. A
/// joint leaf's tampers are refused: a value moved in any of its blocks, two
/// blocks swapped, and a value short.
#[test]
fn a_blake3_base_commits_one_tree_a_run_and_refuses_its_tampers() {
    use multilinear::whir_chain::{GroupFold, Round0Trees};
    let elf = asm_elf_bytes("all_instructions_64");
    let mut one = many_groups(BaseFormat::B3_WHIR);
    one.group = GroupFold::Indexed;
    assert_eq!(one.zf.whir_trees, Round0Trees::One, "the default");
    assert_eq!(one.zf.check_trees(), Ok(()));
    let mut each = one;
    each.zf.whir_trees = Round0Trees::Each;
    crate::base_only::with_permit(|| {
        let honest = prove(&elf, &one, 3);
        let per_tree = prove(&elf, &each, 3);
        assert_eq!(
            refusal(&honest, &elf, &one),
            None,
            "one tree a run verifies"
        );
        assert_eq!(refusal(&per_tree, &elf, &each), None, "each verifies");
        assert!(!verify(&honest, &elf, &each), "a joint proof under each");
        assert!(!verify(&per_tree, &elf, &one), "a per-tree proof under one");
        assert!(
            honest.proof.roots.len() < per_tree.proof.roots.len(),
            "{} roots against {}",
            honest.proof.roots.len(),
            per_tree.proof.roots.len()
        );
        // The first joint round 0: (group, chain) and its leaf width.
        let (g, c, width) = honest
            .proof
            .columns
            .iter()
            .enumerate()
            .find_map(|(g, column)| {
                column.polys.iter().enumerate().find_map(|(c, chain)| {
                    match &chain.rounds[0].openings {
                        RoundOpenings::BaseJoint(p) => Some((g, c, p.current[0].values.len())),
                        _ => None,
                    }
                })
            })
            .expect("a grouped run opens a joint leaf");
        let block = match one.zf.whir_folds {
            multilinear::whir_chain::WhirFolds::First(k0) => 1usize << k0.get(),
            other => panic!("the default folds its first round wider: {other:?}"),
        };
        assert!(
            width >= 2 * block && width.is_multiple_of(block),
            "width {width}"
        );
        fn joint_at(
            p: &mut BlockWhirProof,
            g: usize,
            c: usize,
        ) -> &mut multilinear::whir_round::RoundProof<F, E> {
            match &mut p.proof.columns[g].polys[c].rounds[0].openings {
                RoundOpenings::BaseJoint(p) => p,
                other => panic!("group {g} chain {c} is joint: {other:?}"),
            }
        }
        let tampered = |name: &str, edit: &dyn Fn(&mut BlockWhirProof)| {
            let mut proof = honest.clone();
            edit(&mut proof);
            let why = refusal(&proof, &elf, &one);
            println!("B3 JOINT TAMPER {name}: {why:?}");
            assert!(why.is_some(), "{name}: accepted");
        };
        for at in [0, block - 1, block, width - 1] {
            tampered(&format!("value {at}"), &|p| {
                joint_at(p, g, c).current[0].values[at] += FieldElement::<F>::one();
            });
        }
        tampered("blocks 0 and 1 swapped", &|p| {
            let values = &mut joint_at(p, g, c).current[0].values;
            for t in 0..block {
                values.swap(t, block + t);
            }
        });
        tampered("a value short", &|p| {
            joint_at(p, g, c).current[0].values.pop();
        });
    });
}
