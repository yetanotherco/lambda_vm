//! A block's commit-time out-of-domain answers (`multilinear_block::commit_ood_point`)
//! on the small block of [`super::phase_b_overlap_tests`]: every carried
//! polynomial answers and the proof verifies, under either argue format and
//! every chain form — a chain a polynomial or a run under Keccak, and under
//! 7-round BLAKE3 one round-0 tree a run, alone and with the WHIR round form
//! (the BLAKE3 base's production form); an answer altered, missing, extra or
//! moved to another group is refused; a false answer its chain carries is
//! refused by the chain; and a false answer no chain carries is refused only
//! because the verifier batches the answers into the chains.

use math::field::{
    element::FieldElement, extensions_goldilocks::Degree3GoldilocksExtensionField as Ext,
};
use multilinear::{
    Error as MlError,
    whir_chain::{
        ArgueFormat, ChainConfig, ChainFormat, GroupFold, Round0Trees, RoundForm, RoundOpenings,
        StackVars,
    },
    whir_commit::LeafOrder,
    whir_hash::Blake3Whir,
};

use super::phase_b_overlap_tests::{
    ARGUES, SIZES, config_for, in_order, prove_block, prove_block_under, verify_block,
    verify_block_under,
};
use crate::multilinear_block::{BlockProved, OodFault, OverlapFaults, PhaseBOverlap};
use crate::multilinear_table::batched::VerifierChecks;

type Fp = math::field::goldilocks::GoldilocksField;

/// How the block's stacked polynomials are opened, and under which hash.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Form {
    /// Keccak, one chain a polynomial.
    PerPoly,
    /// Keccak, one chain a run, a round-0 tree a polynomial.
    PerRun,
    /// 7-round BLAKE3, one chain a run, one round-0 tree a run (joint
    /// leaves, coset order): how production commits a group of two or more.
    Blake3OneTree,
    /// [`Self::Blake3OneTree`] under the WHIR round form, its counts derived
    /// for the target: the BLAKE3 base's chains.
    Blake3Production,
}

const FORMS: [Form; 4] = [
    Form::PerPoly,
    Form::PerRun,
    Form::Blake3OneTree,
    Form::Blake3Production,
];

/// The widest chain any group of the small block runs: every round-form
/// count is derived over this union bound.
const ROUND_FORM_VARS: usize = 12;

fn config(argue: ArgueFormat, form: Form) -> ChainConfig {
    let group = match form {
        Form::PerPoly => GroupFold::PerPoly,
        _ => GroupFold::Indexed,
    };
    let base = config_for(argue, group);
    // Stacked polynomials of 2^3 cells, so each group's columns stack into
    // several and `Indexed` cuts runs of two or more (at the default cap
    // every small group is one polynomial and no tree is joint).
    let one_tree = ChainConfig {
        format: ChainFormat {
            round0: Round0Trees::One,
            leaves: LeafOrder::Coset,
            stack: StackVars::new(3).expect("a tested cap"),
            ..base.format
        },
        ..base
    };
    match form {
        Form::PerPoly | Form::PerRun => base,
        Form::Blake3OneTree => one_tree,
        Form::Blake3Production => {
            let mut c = one_tree
                .with_whir_rounds(2, 32, ROUND_FORM_VARS)
                .expect("a tested shrink");
            let RoundForm::Whir(w) = c.format.rounds else {
                unreachable!("the round form")
            };
            c.num_queries = multilinear::query_count::num_queries(
                c.log_blowup,
                w.union_rounds(),
                w.security_bits(),
                c.grind.query,
            );
            c
        }
    }
}

fn blake3(form: Form) -> bool {
    matches!(form, Form::Blake3OneTree | Form::Blake3Production)
}

/// The first group whose first chain is a run of two or more under one
/// tree a run — a joint round-0 leaf — in `form`'s honest proof; group 0
/// under the forms with no joint tree.
fn joint_group(argue: ArgueFormat, form: Form) -> usize {
    if !blake3(form) {
        return 0;
    }
    let honest = prove_at(argue, form, None);
    honest
        .0
        .columns
        .iter()
        .position(|column| {
            matches!(
                column.polys[0].rounds[0].openings,
                RoundOpenings::BaseJoint(_)
            )
        })
        .unwrap_or_else(|| panic!("{argue:?} {form:?}: a group's first chain is a joint run"))
}

/// The block proved, its fault (if any) on `at`'s answer: `(group, poly)`.
fn prove_at(
    argue: ArgueFormat,
    form: Form,
    fault: Option<(OodFault, usize)>,
) -> BlockProved<Fp, Ext> {
    let fault = |block: &mut crate::multilinear_block::BlockCommitted<'_, Fp, Ext>| {
        if let Some((fault, group)) = fault {
            assert!(
                block.fault_commit_ood_at(fault, group, 0),
                "an answer to make false"
            );
        }
    };
    let cfg = config(argue, form);
    let (proved, _) = if blake3(form) {
        prove_block_under::<Blake3Whir>(
            &cfg,
            PhaseBOverlap::Off,
            &in_order(),
            OverlapFaults::default(),
            &fault,
        )
    } else {
        prove_block(
            &cfg,
            PhaseBOverlap::Off,
            &in_order(),
            OverlapFaults::default(),
            &fault,
        )
    };
    proved.unwrap_or_else(|e| panic!("{argue:?} {form:?}: {e:?}"))
}

/// The block proved under `form`; a fault makes the first answer of
/// [`joint_group`] false — a polynomial a joint run carries under BLAKE3.
fn proved(argue: ArgueFormat, form: Form, fault: Option<OodFault>) -> BlockProved<Fp, Ext> {
    prove_at(argue, form, fault.map(|f| (f, joint_group(argue, form))))
}

fn check_with(
    argue: ArgueFormat,
    form: Form,
    proved: &BlockProved<Fp, Ext>,
    answers: &[FieldElement<Ext>],
    checks: VerifierChecks,
) -> Result<(), MlError> {
    let cfg = config(argue, form);
    if blake3(form) {
        verify_block_under::<Blake3Whir>(&cfg, proved, answers, checks)
    } else {
        verify_block(&cfg, proved, answers, checks)
    }
}

fn check(
    argue: ArgueFormat,
    form: Form,
    proved: &BlockProved<Fp, Ext>,
    answers: &[FieldElement<Ext>],
) -> Result<(), MlError> {
    check_with(argue, form, proved, answers, VerifierChecks::ALL)
}

/// ★ Every carried polynomial answers — one per root, in root order — and the
/// block verifies with them, under each argue format and each chain split.
/// The proof carries polynomials enough that `Indexed` cuts a run of two or
/// more somewhere, so a grouped chain carries answers; under BLAKE3 group
/// 0's first chain is a joint run.
#[test]
fn every_carried_polynomial_answers_and_the_block_verifies() {
    for argue in ARGUES {
        for form in FORMS {
            let proved = proved(argue, form, None);
            // Under BLAKE3 a group's first chain is a joint run (or this panics).
            let _ = joint_group(argue, form);
            if !blake3(form) {
                assert_eq!(proved.4.len(), proved.0.roots.len(), "{argue:?} {form:?}");
            } else {
                assert!(
                    proved.4.len() > proved.0.roots.len(),
                    "{argue:?} {form:?}: a joint tree"
                );
            }
            assert_eq!(proved.3.len(), SIZES.len());
            check(argue, form, &proved, &proved.4)
                .unwrap_or_else(|e| panic!("{argue:?} {form:?}: {e:?}"));
        }
    }
}

/// ★ An answer altered, missing, extra, or swapped with another group's is
/// refused: the count before anything is absorbed, the rest at the
/// challenges they moved or at the chain that carries them.
#[test]
fn an_altered_missing_extra_or_moved_answer_is_refused() {
    for argue in ARGUES {
        for form in FORMS {
            let proved = proved(argue, form, None);
            let answers = &proved.4;
            let n = answers.len();
            assert!(n >= 3, "the block carries a few polynomials");
            for at in [0, n / 2, n - 1] {
                let mut altered = answers.clone();
                altered[at] += FieldElement::<Ext>::one();
                assert!(
                    check(argue, form, &proved, &altered).is_err(),
                    "{argue:?} {form:?}: answer {at} altered"
                );
            }
            assert_eq!(
                check(argue, form, &proved, &answers[..n - 1]),
                Err(MlError::CommitOodCount {
                    expected: n,
                    got: n - 1
                }),
                "{argue:?} {form:?}: an answer missing"
            );
            let mut extra = answers.clone();
            extra.push(FieldElement::<Ext>::one());
            assert_eq!(
                check(argue, form, &proved, &extra),
                Err(MlError::CommitOodCount {
                    expected: n,
                    got: n + 1
                }),
                "{argue:?} {form:?}: an answer too many"
            );
            let mut moved = answers.clone();
            moved.swap(0, n - 1);
            assert_ne!(answers[0], answers[n - 1]);
            assert!(
                check(argue, form, &proved, &moved).is_err(),
                "{argue:?} {form:?}: the first group's answer swapped with the last's"
            );
        }
    }
}

/// A prover that answers falsely — and absorbs, sends and carries the false
/// answer, so every challenge is the one it drew — is refused by the chain
/// that carries it.
#[test]
fn a_false_answer_its_chain_carries_is_refused() {
    for argue in ARGUES {
        for form in FORMS {
            let proved = proved(argue, form, Some(OodFault::False));
            let refused = check(argue, form, &proved, &proved.4);
            assert!(
                matches!(refused, Err(MlError::ColumnOpeningRejected { .. })),
                "{argue:?} {form:?}: {refused:?}"
            );
        }
    }
}

/// ★ Mutation: the batching is what binds an answer. A prover whose false
/// answer no chain carries is refused by the verifier; the same verifier with
/// the answers left out of its chains (`VerifierChecks::commit_ood` off)
/// accepts it — the answers are then absorbed and counted, and bind nothing.
#[test]
fn the_batching_is_what_refuses_an_unbound_false_answer() {
    for argue in ARGUES {
        for form in FORMS {
            let proved = proved(argue, form, Some(OodFault::FalseUnbound));
            assert!(
                check(argue, form, &proved, &proved.4).is_err(),
                "{argue:?} {form:?}: refused"
            );
            let unbatched = VerifierChecks {
                commit_ood: false,
                ..VerifierChecks::ALL
            };
            check_with(argue, form, &proved, &proved.4, unbatched).unwrap_or_else(|e| {
                panic!("{argue:?} {form:?}: a verifier that does not batch refused: {e:?}")
            });
        }
    }
}
