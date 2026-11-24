//! Phase B's groups split over two cards (`LAMBDA_VM_TWO_CARD_SPLIT`): each
//! group proves on its own fork of the post-roots transcript and its share is
//! put back by group index, so the split moves no byte.
//!
//! Box tier, `#[ignore]` (a card, each run alone, deterministic grind): a
//! small many-group block proves the same bytes split as on one card, and
//! verifies — forced on one card (a second backend of card 0 drives its
//! share) and, on a two-card box, on card 1, alone and with the two-card
//! openings on as well; one root of the split proof swapped is refused.
//! The split's schedule and assembly are host-tested in the stark crate
//! (`phase_b_split_tests`), its knob in math-cuda (`two_card_split_tests`).

#![cfg(feature = "cuda")]

use crate::block_whir::{BlockWhirProof, Deviations, prove_block_whir_with, verify_block_whir};
use crate::tests::block_whir_p1_tests::{many_groups, options};
use math_cuda::device::{
    TwoCardOpenings, TwoCardSplit, card_count, set_two_card_openings_for_test,
    set_two_card_split_for_test, split_placement,
};
use stark::proof::options::{BaseFormat, ProofOptions};

/// Restores the process's two-card settings when dropped.
struct Restore;

impl Drop for Restore {
    fn drop(&mut self) {
        set_two_card_split_for_test(None);
        set_two_card_openings_for_test(None);
    }
}

fn bytes(p: &BlockWhirProof) -> Vec<u8> {
    rkyv::to_bytes::<rkyv::rancor::Error>(p)
        .expect("serialize")
        .to_vec()
}

fn elf() -> Vec<u8> {
    crate::test_utils::asm_elf_bytes("test_commit_4")
}

/// A small BLAKE3 block over many groups, proved and checked by the host
/// verifier.
fn proved() -> BlockWhirProof {
    let elf = elf();
    let format = many_groups(BaseFormat::B3_WHIR);
    let opts = ProofOptions::default_test_options();
    let (proof, _) = crate::base_only::with_blake3_levels(|| {
        prove_block_whir_with(
            &elf,
            &[],
            &opts,
            &format,
            &options(3),
            &Deviations::default(),
        )
    })
    .expect("the block proves");
    assert!(verifies(&proof), "the proof verifies");
    proof
}

fn verifies(proof: &BlockWhirProof) -> bool {
    let elf = elf();
    let format = many_groups(BaseFormat::B3_WHIR);
    let opts = ProofOptions::default_test_options();
    crate::base_only::with_permit(|| verify_block_whir(proof, &elf, &opts, &format))
        .unwrap_or(false)
}

fn needs_grind_and_card() {
    assert!(
        std::env::var_os("LAMBDA_VM_DETERMINISTIC_GRIND").is_some(),
        "the identity needs the deterministic grind"
    );
    assert!(multilinear::gpu::reserve_budget() > 0, "needs a card");
}

/// ★ Card, one process: run alone, deterministic grind. Forced on one card (a
/// second backend of card 0 drives the second share of the groups, its own
/// streams, staging, ledger and pools), the block proves the same bytes as
/// with the split off, and some group was proved by the second driver.
#[test]
#[ignore = "card: run alone with LAMBDA_VM_DETERMINISTIC_GRIND=1"]
fn forced_split_proves_the_same_bytes_on_one_card() {
    let _restore = Restore;
    needs_grind_and_card();
    set_two_card_openings_for_test(Some(TwoCardOpenings::Off));
    set_two_card_split_for_test(Some(TwoCardSplit::Off));
    let off = bytes(&proved());
    set_two_card_split_for_test(Some(TwoCardSplit::Forced));
    assert!(multilinear::gpu::two_card_split_ready(), "forced: ready");
    let before = stark::multilinear_block::split_second_groups();
    let split = bytes(&proved());
    let second = stark::multilinear_block::split_second_groups() - before;
    println!("TWO-CARD SPLIT forced on one card: {second} group(s) proved by the second driver");
    assert!(second > 0, "no group proved by the second driver");
    assert_eq!(split, off, "the split moved a byte");
}

/// ★ Two-card box, one process: run alone, deterministic grind. With two cards
/// seen and the knob at `auto`, the second driver runs on card 1, and the
/// block proves the same bytes as with the knob off; with the two-card
/// openings on too, the same bytes again.
#[test]
#[ignore = "two cards: run alone with LAMBDA_VM_DETERMINISTIC_GRIND=1"]
fn split_proves_the_same_bytes_on_card_1() {
    let _restore = Restore;
    needs_grind_and_card();
    assert!(card_count() >= 2, "needs two cards, saw {}", card_count());
    set_two_card_openings_for_test(Some(TwoCardOpenings::Off));
    set_two_card_split_for_test(Some(TwoCardSplit::Off));
    let off = bytes(&proved());
    set_two_card_split_for_test(Some(TwoCardSplit::Auto));
    assert_eq!(split_placement().ordinal, 1, "the second driver on card 1");
    assert!(multilinear::gpu::two_card_split_ready(), "card 1 ready");
    let before = stark::multilinear_block::split_second_groups();
    let split = bytes(&proved());
    let second = stark::multilinear_block::split_second_groups() - before;
    println!("TWO-CARD SPLIT on card 1: {second} group(s) proved there");
    assert!(second > 0, "no group proved on card 1");
    assert_eq!(split, off, "card 1 moved a byte");
    set_two_card_openings_for_test(Some(TwoCardOpenings::Auto));
    let both = bytes(&proved());
    assert_eq!(
        both, off,
        "the split with the two-card openings moved a byte"
    );
}

/// ★ Card, one process: run alone, deterministic grind. A split proof with any
/// one root swapped for a different root is refused, each root in turn — so
/// the roots of the groups the second driver proved too (the proof as proved
/// verifies, in [`proved`]). Which groups it proved is stamped, not in the
/// proof; the stark crate's host test swaps a root of exactly such a group.
#[test]
#[ignore = "card: run alone with LAMBDA_VM_DETERMINISTIC_GRIND=1"]
fn a_split_proof_with_a_root_swapped_is_refused() {
    let _restore = Restore;
    needs_grind_and_card();
    set_two_card_split_for_test(Some(if card_count() >= 2 {
        TwoCardSplit::Auto
    } else {
        TwoCardSplit::Forced
    }));
    assert!(multilinear::gpu::two_card_split_ready(), "split ready");
    let before = stark::multilinear_block::split_second_groups();
    let proof = proved();
    assert!(
        stark::multilinear_block::split_second_groups() > before,
        "the proof was split"
    );
    let roots = &proof.proof.roots;
    for k in 0..roots.len() {
        let Some(other) = (0..roots.len()).find(|&j| roots[j] != roots[k]) else {
            continue;
        };
        let mut swapped = proof.clone();
        swapped.proof.roots[k] = roots[other];
        assert!(
            !verifies(&swapped),
            "root {k} swapped for root {other} was accepted"
        );
    }
}
