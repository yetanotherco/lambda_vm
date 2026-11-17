//! The block's base proof under standard 7-round BLAKE3
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
    want.extend_from_slice(b"/C8");
    // The final polynomial's width, the base's own unless the measurement
    // knob turned it off (no test sets it).
    assert_eq!(
        crate::block_whir::b3_base_final(),
        multilinear::whir_chain::FinalPoly::Vars(crate::block_whir::B3_BASE_FINAL_VARS)
    );
    want.extend_from_slice(b"/F9");
    assert_eq!(block_statement_tag(&b3), want);
    // A BLAKE3 format's tag names its group size; Poseidon1's is its base's.
    let at = |base: BaseFormat, group_polys: usize| {
        crate::block_whir::block_format_tag(&BlockFormat {
            zf: ZfFormat::DEFAULT.with_base(base),
            group_polys,
            ..BlockFormat::pinned()
        })
    };
    let mut four = want.clone();
    four.extend_from_slice(b"/G4");
    assert_eq!(at(b3, 4), four);
    let mut three = want.clone();
    three.extend_from_slice(b"/G3");
    assert_eq!(at(b3, 3), three);
    assert_eq!(
        at(BaseFormat::P1_WHIR, 3),
        block_statement_tag(&BaseFormat::P1_WHIR),
        "Poseidon1's tag does not move"
    );
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

/// The BLAKE3 base's final-polynomial knob: unset and its two words are the
/// base's width, its two others turn it off, and anything else is refused.
#[test]
fn the_blake3_base_final_knob_reads_its_words_and_refuses_the_rest() {
    use crate::block_whir::{B3_BASE_FINAL_VARS, b3_base_final_setting};
    use multilinear::whir_chain::FinalPoly;
    let on = FinalPoly::Vars(B3_BASE_FINAL_VARS);
    for (word, want) in [
        (None, on),
        (Some("on"), on),
        (Some("9"), on),
        (Some(" ON "), on),
        (Some("off"), FinalPoly::Off),
        (Some("0"), FinalPoly::Off),
    ] {
        assert_eq!(b3_base_final_setting(word), Ok(want), "{word:?}");
    }
    for word in ["", "1", "6", "10", "yes"] {
        assert!(b3_base_final_setting(Some(word)).is_err(), "{word:?}");
    }
}

/// The BLAKE3 base's group-size knob: unset and `4` are the base's four, `3`
/// is Poseidon1's three, and anything else is refused.
#[test]
fn the_blake3_base_group_knob_reads_its_words_and_refuses_the_rest() {
    use crate::block_whir::{B3_BASE_GROUP_POLYS, b3_base_group_setting};
    assert_eq!(B3_BASE_GROUP_POLYS, 4);
    for (word, want) in [(None, 4), (Some("4"), 4), (Some(" 4 "), 4), (Some("3"), 3)] {
        assert_eq!(b3_base_group_setting(word), Ok(want), "{word:?}");
    }
    for word in ["", "2", "5", "four", "0"] {
        assert!(b3_base_group_setting(Some(word)).is_err(), "{word:?}");
    }
}

/// The BLAKE3 base's cap knob: unset is the base's 2^8, a height from 3 to 8
/// is that height, and anything else is refused.
#[test]
fn the_blake3_base_cap_knob_reads_its_words_and_refuses_the_rest() {
    use crate::block_whir::{B3_BASE_CAP, b3_base_cap, b3_base_cap_setting};
    assert_eq!(b3_base_cap(), B3_BASE_CAP, "no test sets the knob");
    assert_eq!(b3_base_cap_setting(None), Ok(8));
    for c in 3u8..=8 {
        assert_eq!(b3_base_cap_setting(Some(&c.to_string())), Ok(c));
    }
    assert_eq!(b3_base_cap_setting(Some(" 6 ")), Ok(6));
    for word in ["", "2", "9", "0", "eight", "-1"] {
        assert!(b3_base_cap_setting(Some(word)).is_err(), "{word:?}");
    }
}

// ───────────────────────────── the WHIR round form ───────────────────────────

/// The BLAKE3 base's round-form knob: unset, `0` and `fri` are today's form;
/// only `1` and `w3` (the design's shrink 3) or `w2` / `w1` turn the round
/// form on; anything else is refused. No test sets it, and neither the
/// pinned verifier's format nor a production format without it carries the
/// round form.
#[test]
fn the_blake3_base_rounds_knob_reads_its_words_and_refuses_the_rest() {
    use crate::block_whir::{B3Rounds, b3_base_rounds, b3_base_rounds_setting};
    for (word, want) in [
        (None, B3Rounds::Fri),
        (Some("0"), B3Rounds::Fri),
        (Some("fri"), B3Rounds::Fri),
        (Some(" FRI "), B3Rounds::Fri),
        (Some("1"), B3Rounds::WHIR3),
        (Some("w3"), B3Rounds::WHIR3),
        (Some(" W3 "), B3Rounds::WHIR3),
        (
            Some("w2"),
            B3Rounds::Whir {
                shrink: 2,
                caps_by_q: true,
            },
        ),
        (
            Some("w1"),
            B3Rounds::Whir {
                shrink: 1,
                caps_by_q: true,
            },
        ),
    ] {
        assert_eq!(b3_base_rounds_setting(word), Ok(want), "{word:?}");
    }
    for word in ["", "2", "3", "on", "true", "whir", "w0", "w4", "w", "yes"] {
        assert!(b3_base_rounds_setting(Some(word)).is_err(), "{word:?}");
    }
    assert_eq!(b3_base_rounds(), B3Rounds::Fri, "no test sets the knob");
    assert_eq!(BlockFormat::pinned().b3_rounds, B3Rounds::Fri);
    assert_eq!(BlockFormat::production().b3_rounds, B3Rounds::Fri);
    assert_eq!(B3Rounds::Fri.tag_word(), "");
    assert_eq!(B3Rounds::WHIR3.tag_word(), "/W3/Cq");
    assert_eq!(
        B3Rounds::Whir {
            shrink: 3,
            caps_by_q: false
        }
        .tag_word(),
        "/W3"
    );
}

/// Shapes whose stack is the production `2^27` (one table of 2^21 rows and
/// 64 columns, beside a narrow one).
fn production_shapes() -> Vec<(usize, usize)> {
    vec![(64, 21), (8, 20)]
}

/// ★ The round form under the BLAKE3 base is a format of its own: its tag
/// ends `/G4/W3` where today's ends `/G4`, its chains' config is the round
/// form at shrink 3 with caps by Q, sized for the same 128-bit target and
/// union bound round 0's count was (so `check_round_form` takes it), and its
/// fold word is today's with the round form's bits. Under today's form the
/// config carries no round-form bit. A Poseidon1 format ignores the field:
/// its tag and its config do not move.
#[test]
fn the_blake3_round_form_is_named_in_the_tag_and_bound_by_the_fold_word() {
    use crate::block_whir::{B3Rounds, block_format_tag};
    use multilinear::whir_chain::{
        FOLD_WORD_TAG, FOLD_WORD_WHIR, FOLD_WORD_WHIR_CAPS_BY_Q, RoundForm, WhirFolds,
        check_round_form,
    };
    let b3 = BaseFormat::B3_WHIR;
    let fri = BlockFormat {
        zf: ZfFormat::DEFAULT.with_base(b3),
        group_polys: 4,
        ..BlockFormat::pinned()
    };
    let w = BlockFormat {
        b3_rounds: B3Rounds::WHIR3,
        ..fri
    };
    let mut today = block_statement_tag(&b3);
    today.extend_from_slice(b"/G4");
    assert_eq!(block_format_tag(&fri), today);
    let mut want = today.clone();
    want.extend_from_slice(b"/W3/Cq");
    assert_eq!(block_format_tag(&w), want);
    let w2 = BlockFormat {
        b3_rounds: B3Rounds::Whir {
            shrink: 2,
            caps_by_q: true,
        },
        ..fri
    };
    assert_ne!(block_format_tag(&w2), block_format_tag(&w));
    // The cap rule is named beside the round form: the base's cap at every
    // tree is another tag and another fold word.
    let w_policy = BlockFormat {
        b3_rounds: B3Rounds::Whir {
            shrink: 3,
            caps_by_q: false,
        },
        ..fri
    };
    let mut policy_tag = today.clone();
    policy_tag.extend_from_slice(b"/W3");
    assert_eq!(block_format_tag(&w_policy), policy_tag);

    let shapes = production_shapes();
    assert_eq!(
        crate::multilinear_prove::tallest_stack(&fri.zf, &shapes),
        27
    );
    let cf = fri.chain_config(&shapes);
    let cw = w.chain_config(&shapes);
    assert_eq!(cf.format.rounds, RoundForm::Fri);
    assert_eq!(
        cf.fold_word() & (FOLD_WORD_WHIR | FOLD_WORD_WHIR_CAPS_BY_Q),
        0
    );
    let RoundForm::Whir(rounds) = cw.format.rounds else {
        panic!("the round form: {:?}", cw.format.rounds)
    };
    assert_eq!(rounds.shrink(), 3);
    assert_eq!(rounds.security_bits(), 128);
    assert_eq!(rounds.union_rounds(), cf.rounds(27));
    assert!(rounds.caps_by_queries());
    check_round_form(&cw).expect("round 0's count is the target's");
    assert_eq!(cw.num_queries, cf.num_queries, "round 0 queries as today");
    assert_ne!(
        cf.fold_word() & FOLD_WORD_TAG,
        0,
        "today's B3 word is tagged"
    );
    assert_eq!(
        cw.fold_word(),
        cf.fold_word()
            | FOLD_WORD_WHIR
            | (3 << 40)
            | (128 << 24)
            | ((cf.rounds(27) as u64) << 16)
            | FOLD_WORD_WHIR_CAPS_BY_Q
    );
    // Under first fold 6 (the base's default) and first fold 4 (the round
    // form's design pick), the calculator's counts and caps by Q.
    let (r, _) = cw.final_round(27).expect("a final polynomial");
    assert_eq!(&cw.round_queries(27)[..=r], &[229, 57, 46, 39]);
    let mut w4 = w;
    w4.zf.whir_folds = WhirFolds::Uniform;
    let cw4 = w4.chain_config(&shapes);
    check_round_form(&cw4).unwrap();
    let (r4, _) = cw4.final_round(27).expect("a final polynomial");
    assert_eq!(&cw4.round_queries(27)[..=r4], &[229, 114, 76, 57, 46]);
    assert_eq!(&cw4.tree_caps(27)[..=r4], &[8, 7, 7, 6, 6]);

    // A Poseidon1 format with the field set: neither its tag nor its config
    // moves.
    let p1 = BlockFormat::pinned();
    let p1w = BlockFormat {
        b3_rounds: B3Rounds::WHIR3,
        ..p1
    };
    assert_eq!(block_format_tag(&p1w), block_format_tag(&p1));
    assert_eq!(
        p1w.chain_config(&shapes).fold_word(),
        p1.chain_config(&shapes).fold_word()
    );
    assert_eq!(p1w.chain_config(&shapes).format.rounds, RoundForm::Fri);
}

/// ★★ Under the permit a BLAKE3 base under the round form (first fold 4,
/// grouped runs with one joint round-0 tree each, groups retired and revived)
/// proves and verifies on the host; no round that commits a successor opens
/// it, where today's form opens one a query; and each form's verifier
/// refuses the other's proof. Tampers are refused: a later round's opened
/// value, a round's out-of-domain answer, and a successor block slipped into
/// a round of the round form.
#[test]
fn a_blake3_base_under_the_round_form_proves_and_each_form_refuses_the_other() {
    use crate::block_whir::B3Rounds;
    use multilinear::whir_chain::{GroupFold, WhirFolds};
    let elf = asm_elf_bytes("all_instructions_64");
    let mut fri = many_groups(BaseFormat::B3_WHIR);
    fri.group = GroupFold::Indexed;
    fri.zf.whir_folds = WhirFolds::Uniform;
    let w = BlockFormat {
        b3_rounds: B3Rounds::WHIR3,
        ..fri
    };
    crate::base_only::with_permit(|| {
        let honest = prove(&elf, &w, 3);
        let today = prove(&elf, &fri, 3);
        assert_eq!(refusal(&honest, &elf, &w), None, "the round form verifies");
        assert_eq!(refusal(&today, &elf, &fri), None, "today's verifies");
        assert!(
            !verify(&honest, &elf, &fri),
            "a round-form proof under today's"
        );
        assert!(
            !verify(&today, &elf, &w),
            "today's proof under the round form"
        );
        let successor_blocks = |p: &BlockWhirProof| -> (usize, usize) {
            let mut rounds = 0;
            let mut blocks = 0;
            for column in &p.proof.columns {
                for chain in &column.polys {
                    for round in &chain.rounds {
                        if round.next_root.is_some() {
                            rounds += 1;
                            blocks += match &round.openings {
                                RoundOpenings::Base(r) => r.next.len(),
                                RoundOpenings::Extension(r) => r.next.len(),
                                RoundOpenings::BaseGrouped(r) => r.next.len(),
                                RoundOpenings::BaseJoint(r) => r.next.len(),
                            };
                        }
                    }
                }
            }
            (rounds, blocks)
        };
        let (w_rounds, w_blocks) = successor_blocks(&honest);
        let (f_rounds, f_blocks) = successor_blocks(&today);
        println!(
            "B3 ROUND FORM: successor rounds {w_rounds} (blocks {w_blocks}) against today's \
             {f_rounds} (blocks {f_blocks})"
        );
        assert!(w_rounds > 0 && w_blocks == 0, "no successor opened");
        assert!(f_blocks > 0, "today's form opens its successors");

        let tampered = |name: &str, edit: &dyn Fn(&mut BlockWhirProof)| {
            let mut proof = honest.clone();
            edit(&mut proof);
            let why = refusal(&proof, &elf, &w);
            println!("B3 ROUND FORM TAMPER {name}: {why:?}");
            assert!(why.is_some(), "{name}: accepted");
        };
        // The first chain with a round after round 0 that opens its tree.
        let (g, c, r) = honest
            .proof
            .columns
            .iter()
            .enumerate()
            .find_map(|(g, column)| {
                column.polys.iter().enumerate().find_map(|(c, chain)| {
                    chain
                        .rounds
                        .iter()
                        .enumerate()
                        .skip(1)
                        .find(|(_, round)| match &round.openings {
                            RoundOpenings::Extension(p) => !p.current.is_empty(),
                            _ => false,
                        })
                        .map(|(r, _)| (g, c, r))
                })
            })
            .expect("a later round opens its tree");
        tampered(
            "a later round's opened value",
            &|p| match &mut p.proof.columns[g].polys[c].rounds[r].openings {
                RoundOpenings::Extension(p) => p.current[0].values[0] += FieldElement::<E>::one(),
                other => panic!("an extension round: {other:?}"),
            },
        );
        tampered("round 0's out-of-domain answer", &|p| {
            let round = &mut p.proof.columns[g].polys[c].rounds[0];
            let value = round.ood_value.as_mut().expect("round 0 has a successor");
            *value += FieldElement::<E>::one();
        });
        tampered(
            "a successor block slipped in",
            &|p| match &mut p.proof.columns[g].polys[c].rounds[r].openings {
                RoundOpenings::Extension(p) => {
                    let extra = p.current[0].clone();
                    p.next.push(extra);
                }
                other => panic!("an extension round: {other:?}"),
            },
        );
    });
}

/// ★ The cap rule is a verifier constant: a BLAKE3 base under the round form
/// with caps by Q (`/W3/Cq`, production) and one with the base's cap at every
/// tree (`/W3`) each verify under their own format and are refused under the
/// other's. And the trees' drop is the prover's alone: the same block under
/// the round form at first fold 4 (one joint round-0 tree a run) proved with 3,
/// 5, 8 and 10 dropped levels verifies, and is the same proof byte for byte
/// when the grind is deterministic (`LAMBDA_VM_DETERMINISTIC_GRIND`, read once
/// per process: run this test alone with it to see the bytes; otherwise a
/// parallel grind's nonces differ between any two proves).
#[test]
fn a_blake3_round_form_proof_under_another_cap_rule_is_refused_and_the_drop_moves_no_byte() {
    use crate::block_whir::B3Rounds;
    use multilinear::whir_chain::{GroupFold, WhirFolds};
    let elf = asm_elf_bytes("all_instructions_64");
    let mut base = many_groups(BaseFormat::B3_WHIR);
    base.group = GroupFold::Indexed;
    base.zf.whir_folds = WhirFolds::Uniform;
    let by_q = BlockFormat {
        b3_rounds: B3Rounds::WHIR3,
        ..base
    };
    let policy = BlockFormat {
        b3_rounds: B3Rounds::Whir {
            shrink: 3,
            caps_by_q: false,
        },
        ..base
    };
    let bytes = |p: &BlockWhirProof| {
        rkyv::to_bytes::<rkyv::rancor::Error>(p)
            .expect("serialize")
            .to_vec()
    };
    crate::base_only::with_permit(|| {
        let q = prove(&elf, &by_q, 3);
        let p = prove(&elf, &policy, 3);
        assert_eq!(refusal(&q, &elf, &by_q), None, "caps by Q verifies");
        assert_eq!(refusal(&p, &elf, &policy), None, "the base's caps verify");
        assert!(
            !verify(&q, &elf, &policy),
            "a caps-by-Q proof under the base's caps"
        );
        assert!(
            !verify(&p, &elf, &by_q),
            "a base's-caps proof under caps by Q"
        );

        let reference = bytes(&q);
        for drop in [5usize, 8, 10] {
            let other = prove(&elf, &by_q, drop);
            assert_eq!(refusal(&other, &elf, &by_q), None, "drop {drop} verifies");
            if crypto::grinding::deterministic() {
                assert_eq!(bytes(&other), reference, "drop {drop} against drop 3");
                println!("B3 ROUND FORM DROP {drop}: the same bytes as drop 3");
            }
        }
    });
}

/// The card's drop under the format: today's 8 at the default (first fold 6),
/// and under a first fold of 4 or 5 enough levels that a `2^27` stack's
/// round-0 cosets stay within the card's coset plan (`k0 + dropped ≥ 27 −
/// 13`); the host's 4 throughout. A prover's choice: no proof byte moves.
#[test]
fn the_card_drop_keeps_round_zero_cosets_within_the_cards_plan() {
    use crate::block_whir::{
        BLOCK_TREE_DROP_LEVELS, BLOCK_TREE_DROP_LEVELS_DEVICE, CARD_COSET_FOLD_LEVELS,
        tree_drop_for,
    };
    use multilinear::whir_chain::{FirstFold, StackVars, WhirFolds};
    let at = |folds: WhirFolds, stack: usize| {
        let mut zf = ZfFormat::DEFAULT;
        zf.whir_folds = folds;
        zf.whir_stack = StackVars::new(stack).unwrap();
        tree_drop_for(&zf)
    };
    let first = |k: usize| WhirFolds::First(FirstFold::new(k).unwrap());
    assert_eq!(
        tree_drop_for(&ZfFormat::DEFAULT).device,
        BLOCK_TREE_DROP_LEVELS_DEVICE
    );
    assert_eq!(BLOCK_TREE_DROP_LEVELS_DEVICE, 8);
    assert_eq!(at(first(6), 27).device, 8);
    assert_eq!(at(first(5), 27).device, 9);
    assert_eq!(at(WhirFolds::Uniform, 27).device, 10);
    assert_eq!(at(WhirFolds::Uniform, 25).device, 8);
    for folds in [first(5), first(6), WhirFolds::Uniform] {
        for stack in 20..=27 {
            let drop = at(folds, stack);
            let k0 = match folds {
                WhirFolds::Uniform => 4,
                WhirFolds::First(k) => k.get(),
            };
            assert!(
                k0 + drop.device + CARD_COSET_FOLD_LEVELS >= stack,
                "{folds:?} {stack}"
            );
            assert_eq!(drop.host, BLOCK_TREE_DROP_LEVELS);
        }
    }
}
