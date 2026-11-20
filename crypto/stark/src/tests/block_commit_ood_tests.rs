//! A block's commit-time out-of-domain answers (`multilinear_block::commit_ood_point`)
//! on the small block of [`super::phase_b_overlap_tests`]: every carried
//! polynomial answers and the proof verifies, under either argue format and
//! either chain split; an answer altered, missing, extra or moved to another
//! group is refused; a false answer its chain carries is refused by the
//! chain; and a false answer no chain carries is refused only because the
//! verifier batches the answers into the chains.

use math::field::{
    element::FieldElement, extensions_goldilocks::Degree3GoldilocksExtensionField as Ext,
};
use multilinear::{
    Error as MlError,
    whir_chain::{ArgueFormat, GroupFold},
};

use super::phase_b_overlap_tests::{
    ARGUES, SIZES, config_for, in_order, prove_block, verify_block,
};
use crate::multilinear_block::{BlockProved, OodFault, OverlapFaults, PhaseBOverlap};
use crate::multilinear_table::batched::VerifierChecks;

type Fp = math::field::goldilocks::GoldilocksField;

const GROUPS: [GroupFold; 2] = [GroupFold::PerPoly, GroupFold::Indexed];

fn proved(argue: ArgueFormat, group: GroupFold, fault: Option<OodFault>) -> BlockProved<Fp, Ext> {
    let (proved, _) = prove_block(
        &config_for(argue, group),
        PhaseBOverlap::Off,
        &in_order(),
        OverlapFaults::default(),
        &|block| {
            if let Some(fault) = fault {
                assert!(block.fault_commit_ood(fault), "an answer to make false");
            }
        },
    );
    proved.unwrap_or_else(|e| panic!("{argue:?} {group:?}: {e:?}"))
}

fn check(
    argue: ArgueFormat,
    group: GroupFold,
    proved: &BlockProved<Fp, Ext>,
    answers: &[FieldElement<Ext>],
) -> Result<(), MlError> {
    verify_block(
        &config_for(argue, group),
        proved,
        answers,
        VerifierChecks::ALL,
    )
}

/// ★ Every carried polynomial answers — one per root, in root order — and the
/// block verifies with them, under each argue format and each chain split.
/// The proof carries polynomials enough that `Indexed` cuts a run of two or
/// more somewhere, so a grouped chain carries answers.
#[test]
fn every_carried_polynomial_answers_and_the_block_verifies() {
    for argue in ARGUES {
        for group in GROUPS {
            let proved = proved(argue, group, None);
            assert_eq!(proved.4.len(), proved.0.roots.len(), "{argue:?} {group:?}");
            assert_eq!(proved.3.len(), SIZES.len());
            check(argue, group, &proved, &proved.4)
                .unwrap_or_else(|e| panic!("{argue:?} {group:?}: {e:?}"));
        }
    }
}

/// ★ An answer altered, missing, extra, or swapped with another group's is
/// refused: the count before anything is absorbed, the rest at the
/// challenges they moved or at the chain that carries them.
#[test]
fn an_altered_missing_extra_or_moved_answer_is_refused() {
    for argue in ARGUES {
        for group in GROUPS {
            let proved = proved(argue, group, None);
            let answers = &proved.4;
            let n = answers.len();
            assert!(n >= 3, "the block carries a few polynomials");
            for at in [0, n / 2, n - 1] {
                let mut altered = answers.clone();
                altered[at] += FieldElement::<Ext>::one();
                assert!(
                    check(argue, group, &proved, &altered).is_err(),
                    "{argue:?} {group:?}: answer {at} altered"
                );
            }
            assert_eq!(
                check(argue, group, &proved, &answers[..n - 1]),
                Err(MlError::CommitOodCount {
                    expected: n,
                    got: n - 1
                }),
                "{argue:?} {group:?}: an answer missing"
            );
            let mut extra = answers.clone();
            extra.push(FieldElement::<Ext>::one());
            assert_eq!(
                check(argue, group, &proved, &extra),
                Err(MlError::CommitOodCount {
                    expected: n,
                    got: n + 1
                }),
                "{argue:?} {group:?}: an answer too many"
            );
            let mut moved = answers.clone();
            moved.swap(0, n - 1);
            assert_ne!(answers[0], answers[n - 1]);
            assert!(
                check(argue, group, &proved, &moved).is_err(),
                "{argue:?} {group:?}: the first group's answer swapped with the last's"
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
        for group in GROUPS {
            let proved = proved(argue, group, Some(OodFault::False));
            let refused = check(argue, group, &proved, &proved.4);
            assert!(
                matches!(refused, Err(MlError::ColumnOpeningRejected { .. })),
                "{argue:?} {group:?}: {refused:?}"
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
        for group in GROUPS {
            let proved = proved(argue, group, Some(OodFault::FalseUnbound));
            assert!(
                check(argue, group, &proved, &proved.4).is_err(),
                "{argue:?} {group:?}: refused"
            );
            let unbatched = VerifierChecks {
                commit_ood: false,
                ..VerifierChecks::ALL
            };
            verify_block(&config_for(argue, group), &proved, &proved.4, unbatched).unwrap_or_else(
                |e| panic!("{argue:?} {group:?}: a verifier that does not batch refused: {e:?}"),
            );
        }
    }
}
