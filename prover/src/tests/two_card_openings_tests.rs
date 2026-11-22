//! Phase B's openings on a second card (`LAMBDA_VM_TWO_CARD_OPENINGS`): each
//! group proves on its own fork, so where its openings run moves no byte.
//!
//! Host: without a card (or off) no group is sent to another card and phase B
//! is today's. Box tier, `#[ignore]` (a card, run alone, deterministic grind):
//! a small many-group block proves the same bytes with the openings on their
//! own backend as on card 0's, and verifies — forced on one card (a second
//! backend of card 0) and, on a two-card box, on card 1.

use crate::block_whir::{BlockWhirProof, Deviations, prove_block_whir_with, verify_block_whir};
use crate::tests::block_whir_p1_tests::{many_groups, options};
use stark::proof::options::{BaseFormat, ProofOptions};

/// Restores the process's two-card setting when dropped.
#[cfg(feature = "cuda")]
struct Restore;

#[cfg(feature = "cuda")]
impl Drop for Restore {
    fn drop(&mut self) {
        math_cuda::device::set_two_card_openings_for_test(None);
    }
}

fn bytes(p: &BlockWhirProof) -> Vec<u8> {
    rkyv::to_bytes::<rkyv::rancor::Error>(p)
        .expect("serialize")
        .to_vec()
}

/// A small BLAKE3 block over many groups (several groups open beside an
/// argue), proved and checked by the host verifier: its bytes.
fn proved() -> Vec<u8> {
    let elf = crate::test_utils::asm_elf_bytes("test_commit_4");
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
    assert!(
        crate::base_only::with_permit(|| verify_block_whir(&proof, &elf, &opts, &format))
            .expect("verify runs"),
        "the proof verifies"
    );
    bytes(&proof)
}

/// Without a device build, or with the knob off, no group's openings leave
/// card 0's path: `two_card_openings` reads false and none is counted.
#[test]
fn off_or_without_a_card_no_group_opens_elsewhere() {
    #[cfg(feature = "cuda")]
    let _restore = Restore;
    #[cfg(feature = "cuda")]
    math_cuda::device::set_two_card_openings_for_test(Some(
        math_cuda::device::TwoCardOpenings::Off,
    ));
    assert!(!multilinear::gpu::two_card_openings());
    let before = stark::multilinear_block::two_card_opened();
    let _ = proved();
    assert_eq!(
        stark::multilinear_block::two_card_opened(),
        before,
        "a group opened on another card with the two-card openings off"
    );
}

/// ★ Card, one process: run alone, deterministic grind. The block proves the
/// same bytes with the two-card openings off and forced (one card: a second
/// backend of card 0 for the openings, its own streams, staging and ledger),
/// and some group's openings ran there.
#[cfg(feature = "cuda")]
#[test]
#[ignore = "card: run alone with LAMBDA_VM_DETERMINISTIC_GRIND=1"]
fn forced_two_card_openings_prove_the_same_bytes_on_one_card() {
    use math_cuda::device::{TwoCardOpenings, set_two_card_openings_for_test};
    let _restore = Restore;
    assert!(
        std::env::var_os("LAMBDA_VM_DETERMINISTIC_GRIND").is_some(),
        "the identity needs the deterministic grind"
    );
    assert!(multilinear::gpu::reserve_budget() > 0, "needs a card");
    set_two_card_openings_for_test(Some(TwoCardOpenings::Off));
    let off = proved();
    set_two_card_openings_for_test(Some(TwoCardOpenings::Forced));
    assert!(multilinear::gpu::two_card_openings(), "forced on");
    let before = stark::multilinear_block::two_card_opened();
    let forced = proved();
    let opened = stark::multilinear_block::two_card_opened() - before;
    println!(
        "TWO-CARD OPENINGS forced on one card: {opened} group(s) opened on the openings' backend"
    );
    assert!(opened > 0, "no group opened on the openings' backend");
    assert_eq!(forced, off, "the openings' backend moved a byte");
}

/// ★ Two-card box, one process: run alone, deterministic grind. With two cards
/// seen and the knob at `auto`, the openings run on card 1, and
/// the block proves the same bytes as with the knob off.
#[cfg(feature = "cuda")]
#[test]
#[ignore = "two cards: run alone with LAMBDA_VM_DETERMINISTIC_GRIND=1"]
fn two_card_openings_prove_the_same_bytes_on_card_1() {
    use math_cuda::device::{
        TwoCardOpenings, card_count, openings_placement, set_two_card_openings_for_test,
    };
    let _restore = Restore;
    assert!(
        std::env::var_os("LAMBDA_VM_DETERMINISTIC_GRIND").is_some(),
        "the identity needs the deterministic grind"
    );
    assert!(card_count() >= 2, "needs two cards, saw {}", card_count());
    set_two_card_openings_for_test(Some(TwoCardOpenings::Off));
    let off = proved();
    set_two_card_openings_for_test(Some(TwoCardOpenings::Auto));
    assert_eq!(openings_placement().ordinal, 1, "the openings on card 1");
    let before = stark::multilinear_block::two_card_opened();
    let two = proved();
    let opened = stark::multilinear_block::two_card_opened() - before;
    println!("TWO-CARD OPENINGS on card 1: {opened} group(s) opened there");
    assert!(opened > 0, "no group opened on card 1");
    assert_eq!(two, off, "card 1 moved a byte");
}
