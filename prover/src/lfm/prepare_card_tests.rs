//! A W-LFM build that keeps its prepared tables row-major and has the card
//! transpose them at the commit (`LAMBDA_VM_LEVELS_PREPARE_ON_CARD`) is the
//! build that gathers them on the host: the same prepared columns, the same
//! chip prefixes packed from them, the same identity (so the same roots), and
//! the same proof bytes, in every stack place. Without a card (the laptop) the
//! card places fall back to the host's gather, which pins the row-major
//! stacks' grouping, the prefix packing and the fallback itself; the card's
//! transpose and its read-back are the box tier's, with a fault in the
//! transpose shown to move the identity.

use super::compiler::LfmProgram;
use super::hash::HasherKind;
use super::wlfm::{
    StackPlace, WlfmBuild, WlfmFamily, WlfmLevers, WlfmRate, prepare_wlfm, prove_wlfm,
    set_prepare_on_card_for_test, test_format, verify_wlfm,
};

/// Restores this thread's override (and the card's fault) when it drops.
struct Restore;

impl Drop for Restore {
    fn drop(&mut self) {
        set_prepare_on_card_for_test(None);
        multilinear::gpu::fault_row_major_for_tests(false);
    }
}

fn bytes(proof: &super::wlfm::WlfmProof) -> Vec<u8> {
    rkyv::to_bytes::<rkyv::rancor::Error>(proof)
        .expect("a W-LFM proof serializes")
        .to_vec()
}

/// `program`'s build in `place`, its tables kept row-major or not, its chip
/// prefixes packed before the commit as the levels' builds pack them.
fn built(
    program: &LfmProgram,
    hasher: HasherKind,
    family: WlfmFamily,
    place: StackPlace,
    on_card: bool,
) -> WlfmBuild {
    let _restore = Restore;
    set_prepare_on_card_for_test(Some(on_card));
    let mut pre = prepare_wlfm(
        program,
        hasher,
        WlfmRate::Two,
        WlfmLevers::with_family(family),
    )
    .unwrap_or_else(|e| panic!("the {place:?} prepare (row-major {on_card}): {e}"));
    pre.hold_chip_prefixes()
        .unwrap_or_else(|e| panic!("the {place:?} prefixes (row-major {on_card}): {e}"));
    pre.commit(place)
        .unwrap_or_else(|e| panic!("the {place:?} commit (row-major {on_card}): {e}"))
}

/// The row-major build is the gathered one in every place; under `card`, a
/// card place must have transposed on the card.
fn the_same_build(
    program: &LfmProgram,
    hasher: HasherKind,
    family: WlfmFamily,
    arenas: &[Vec<super::word::LfmWord>],
    card: bool,
) {
    for place in [StackPlace::Card, StackPlace::CardRetired, StackPlace::Host] {
        let want = built(program, hasher, family, place, false);
        let got = built(program, hasher, family, place, true);
        assert!(
            !want.artifacts.stacks.is_empty(),
            "the program has prepared stacks"
        );
        assert_eq!(
            want.card_transpose(),
            None,
            "{place:?}: the gathered build never transposes"
        );
        if card && place != StackPlace::Host {
            assert!(
                got.card_transpose().is_some(),
                "{place:?}: the card transposed the row-major stacks"
            );
        } else {
            assert_eq!(
                got.card_transpose(),
                None,
                "{place:?}: gathered on the host"
            );
        }
        assert_eq!(
            got.artifacts.program_id, want.artifacts.program_id,
            "{place:?}: the same identity (the same roots)"
        );
        assert!(
            got.prepared_columns() == want.prepared_columns(),
            "{place:?}: the same prepared columns on the host"
        );
        assert!(
            got.chip_prefixes() == want.chip_prefixes(),
            "{place:?}: the same chip prefixes"
        );
        let a = prove_wlfm(program, &want, arenas).expect("the gathered build proves");
        let b = prove_wlfm(program, &got, arenas).expect("the row-major build proves");
        verify_wlfm(&got.artifacts, &b, &b.public_words).expect("and verifies");
        assert!(bytes(&a) == bytes(&b), "{place:?}: the same proof bytes");
    }
}

/// ★ The Poseidon1 fixture: the row-major build is the gathered one in every
/// place (on the laptop, through the card places' host fallback).
#[test]
fn a_row_major_build_is_the_gathered_one_under_poseidon1() {
    let _unground = test_format::set(None);
    the_same_build(
        &super::wlfm_tests::honest(),
        super::wlfm_tests::HASHER,
        WlfmFamily::P1,
        &super::wlfm_tests::arenas(),
        false,
    );
}

/// ★ The lean BLAKE3 fixture under the BLAKE3 family: the same, with the
/// lean round tables' row-major groups among the stacks.
#[test]
fn a_row_major_build_is_the_gathered_one_under_blake3() {
    let _fast = test_format::set(Some(2));
    the_same_build(
        &super::b3l_wlfm_tests::honest(),
        super::wlfm_tests::HASHER,
        WlfmFamily::B3,
        &[],
        false,
    );
}

/// ★ On the card: the stacks cross the bus row-major, the card transposes
/// them, commits from them and the host keeps them read back; the build is the
/// gathered one in every place, byte for byte.
#[test]
#[ignore = "box tier: needs the card"]
fn a_card_transposed_build_is_the_gathered_one() {
    let _fast = test_format::set(Some(2));
    the_same_build(
        &super::b3l_wlfm_tests::honest(),
        super::wlfm_tests::HASHER,
        WlfmFamily::B3,
        &[],
        true,
    );
    let _unground = test_format::set(None);
    the_same_build(
        &super::wlfm_tests::honest(),
        super::wlfm_tests::HASHER,
        WlfmFamily::P1,
        &super::wlfm_tests::arenas(),
        true,
    );
}

/// ⛔ The comparison sees a transpose one word off: with the card's transposed
/// column 0 moved by one at row 0, the row-major build's identity and columns
/// are no longer the gathered build's.
#[test]
#[ignore = "box tier: needs the card"]
fn a_transpose_one_word_off_builds_another_identity() {
    let _unground = test_format::set(None);
    let program = super::wlfm_tests::honest();
    let want = built(
        &program,
        super::wlfm_tests::HASHER,
        WlfmFamily::P1,
        StackPlace::Card,
        false,
    );
    let _restore = Restore;
    multilinear::gpu::fault_row_major_for_tests(true);
    let got = built(
        &program,
        super::wlfm_tests::HASHER,
        WlfmFamily::P1,
        StackPlace::Card,
        true,
    );
    multilinear::gpu::fault_row_major_for_tests(false);
    assert!(got.card_transpose().is_some(), "the card transposed");
    assert_ne!(
        got.artifacts.program_id, want.artifacts.program_id,
        "a moved word moves the identity"
    );
    assert!(
        got.prepared_columns() != want.prepared_columns(),
        "and the host's read-back columns"
    );
}
