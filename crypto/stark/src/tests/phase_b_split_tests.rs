//! Phase B's groups over two cards ([`BlockCommitted::with_two_card_split`],
//! `LAMBDA_VM_TWO_CARD_SPLIT`) on the host: the two drivers run without a
//! second card ([`BlockCommitted::set_two_card_split`]), so the split's
//! schedule, its per-card transcript copies and the shares' assembly by
//! group index are checked here; the second card's device is the box's.
//!
//! The block is [`super::phase_b_overlap_tests`]'s: seven groups, two of them
//! with prepared stacks.

use multilinear::{Error as MlError, whir_chain::ArgueFormat};

use super::phase_b_overlap_tests::{
    ARGUES, BEFORE_THE_OVERLAP, H, Heard, SIZES, bytes, config, digest, in_order, prove_block_with,
    verify,
};
use crate::multilinear_block::{BlockCommitted, BlockProved, OverlapFaults, PhaseBOverlap};
use math::field::{
    extensions_goldilocks::Degree3GoldilocksExtensionField as Ext,
    goldilocks::GoldilocksField as Fp,
};

/// How phase B is asked to split: not at all, opted in (the knob and the
/// card decide: on the host, nothing splits), or two drivers on the host.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Split {
    None,
    OptedIn,
    OnHost,
}

/// The block proved with `split` (groups marked in `keep` stay on card 0),
/// under `argue` and `overlap`, its prepared stacks held or `retired`, its
/// groups in `order`, with a test's `faults`.
fn prove_split(
    argue: ArgueFormat,
    overlap: PhaseBOverlap,
    order: &[usize],
    retired: bool,
    split: Split,
    keep: &[bool],
    faults: OverlapFaults,
) -> (Result<BlockProved<Fp, Ext>, MlError>, Vec<Heard>) {
    let keep = keep.to_vec();
    let set = move |block: &mut BlockCommitted<'_, Fp, Ext>| match split {
        Split::None => {}
        Split::OptedIn => block.set_two_card_split(Some(keep.clone()), false),
        Split::OnHost => block.set_two_card_split(Some(keep.clone()), true),
    };
    prove_block_with::<H>(
        &config(argue),
        overlap,
        order,
        faults,
        &set,
        None,
        retired,
        None,
    )
}

fn cards(proved: &BlockProved<Fp, Ext>) -> Vec<u8> {
    proved.3.iter().map(|s| s.card).collect()
}

fn sorted(mut heard: Vec<Heard>) -> Vec<Heard> {
    heard.sort();
    heard
}

/// ★ Two drivers prove the bytes of one: under either argue format, with the
/// overlap on and off, the prepared stacks held or retired, and the groups in
/// order or reversed, the block split over two drivers proves the one-driver
/// proof byte for byte (in group order and with the stacks held, the pinned
/// bytes of one group at a time), it verifies, both drivers proved groups,
/// and the observer heard every group's same share once.
#[test]
fn two_drivers_prove_the_bytes_of_one() {
    let n = SIZES.len();
    let reversed: Vec<usize> = (0..n).rev().collect();
    let keep = vec![false; n];
    for (argue, pinned) in ARGUES.into_iter().zip(BEFORE_THE_OVERLAP) {
        for retired in [false, true] {
            for overlap in [PhaseBOverlap::Off, PhaseBOverlap::On] {
                for order in [in_order(), reversed.clone()] {
                    let what = format!("{argue:?} retired {retired} {overlap:?} {order:?}");
                    let (one, heard_one) = prove_split(
                        argue,
                        overlap,
                        &order,
                        retired,
                        Split::None,
                        &keep,
                        OverlapFaults::default(),
                    );
                    let one = one.unwrap_or_else(|e| panic!("{what}: one driver: {e:?}"));
                    let (two, heard_two) = prove_split(
                        argue,
                        overlap,
                        &order,
                        retired,
                        Split::OnHost,
                        &keep,
                        OverlapFaults::default(),
                    );
                    let two = two.unwrap_or_else(|e| panic!("{what}: two drivers: {e:?}"));
                    assert!(cards(&one).iter().all(|&c| c == 0), "{what}: one driver");
                    let on = cards(&two);
                    assert!(
                        on.contains(&0) && on.contains(&1),
                        "{what}: both drivers proved groups: {on:?}"
                    );
                    assert_eq!(bytes(&two), bytes(&one), "{what}: two drivers moved a byte");
                    if !retired && order == in_order() && overlap == PhaseBOverlap::Off {
                        assert_eq!(digest(&two), pinned, "{what}: the pinned bytes");
                    }
                    verify(argue, &two).unwrap_or_else(|e| panic!("{what}: verifies: {e:?}"));
                    assert_eq!(heard_two.len(), n, "{what}: every group heard once");
                    assert_eq!(
                        sorted(heard_two),
                        sorted(heard_one),
                        "{what}: the shares heard"
                    );
                }
            }
        }
    }
}

/// A held prepared stack's group, and a group the caller keeps, prove on card
/// 0; the bytes are the same.
#[test]
fn held_and_kept_groups_stay_on_card_0() {
    let n = SIZES.len();
    let mut keep = vec![false; n];
    keep[3] = true;
    keep[4] = true;
    for argue in ARGUES {
        let (one, _) = prove_split(
            argue,
            PhaseBOverlap::On,
            &in_order(),
            false,
            Split::None,
            &keep,
            OverlapFaults::default(),
        );
        let (two, _) = prove_split(
            argue,
            PhaseBOverlap::On,
            &in_order(),
            false,
            Split::OnHost,
            &keep,
            OverlapFaults::default(),
        );
        let (one, two) = (one.expect("one driver"), two.expect("two drivers"));
        let on = cards(&two);
        // Groups 1 and 5 hold prepared stacks (held, not retired); 3 and 4
        // are kept.
        for g in [1, 3, 4, 5] {
            assert_eq!(on[g], 0, "{argue:?}: group {g} left card 0: {on:?}");
        }
        assert!(on.contains(&1), "{argue:?}: nothing split: {on:?}");
        assert_eq!(bytes(&two), bytes(&one), "{argue:?}");
    }
}

/// Opted in, without a second card (the host here; on a card box, the knob
/// unset): phase B runs one driver, every group on card 0, the same bytes.
#[test]
fn without_a_second_card_nothing_splits() {
    let keep = vec![false; SIZES.len()];
    for argue in ARGUES {
        let (one, _) = prove_split(
            argue,
            PhaseBOverlap::On,
            &in_order(),
            true,
            Split::None,
            &keep,
            OverlapFaults::default(),
        );
        let (opted, _) = prove_split(
            argue,
            PhaseBOverlap::On,
            &in_order(),
            true,
            Split::OptedIn,
            &keep,
            OverlapFaults::default(),
        );
        let (one, opted) = (one.expect("one driver"), opted.expect("opted in"));
        if multilinear::gpu::two_card_split_ready() {
            // A card box with the knob on: the split's own card test covers it.
            continue;
        }
        assert!(cards(&opted).iter().all(|&c| c == 0), "{argue:?}");
        assert_eq!(bytes(&opted), bytes(&one), "{argue:?}");
    }
}

/// ★ Refused: in a block split over two drivers, one root of a group the
/// second driver proved, swapped for another group's root, is refused by the
/// verifier (and the proof as proved verifies).
#[test]
fn a_swapped_root_of_a_group_from_the_second_card_is_refused() {
    let n = SIZES.len();
    for argue in ARGUES {
        let (two, _) = prove_split(
            argue,
            PhaseBOverlap::On,
            &in_order(),
            true,
            Split::OnHost,
            &vec![false; n],
            OverlapFaults::default(),
        );
        let two = two.expect("two drivers");
        verify(argue, &two).expect("as proved, it verifies");
        let cfg = config(argue);
        // Each group's first root: the roots block holds each group's round-0
        // trees in group order.
        let mut first_root = Vec::with_capacity(n);
        let mut at = 0usize;
        for stamps in &two.3 {
            first_root.push(at);
            at += multilinear::stacked_eval::round0_trees(stamps.polys, &cfg).len();
        }
        assert_eq!(at, two.0.roots.len(), "{argue:?}: the roots block's length");
        let on = cards(&two);
        let g = (0..n).find(|&g| on[g] == 1).expect("a group on card 1");
        let other = (0..n)
            .find(|&h| two.0.roots[first_root[h]] != two.0.roots[first_root[g]])
            .expect("another root");
        let mut swapped = two.clone();
        swapped.0.roots[first_root[g]] = two.0.roots[first_root[other]];
        assert!(
            verify(argue, &swapped).is_err(),
            "{argue:?}: group {g}'s root swapped for group {other}'s was accepted"
        );
    }
}

/// A failed argue on either driver is the prove's error (never the other
/// driver's stop, never a panic).
#[test]
fn a_failed_argue_on_either_driver_is_the_proves_error() {
    let n = SIZES.len();
    for place in 0..n {
        let (proved, _) = prove_split(
            ArgueFormat::PerTable,
            PhaseBOverlap::On,
            &in_order(),
            true,
            Split::OnHost,
            &vec![false; n],
            OverlapFaults {
                argue_fails: Some(place),
                ..OverlapFaults::default()
            },
        );
        match proved {
            Err(MlError::DeviceFailed { stage }) => {
                assert_eq!(stage, "a test's failed argue", "place {place}")
            }
            other => panic!("place {place}: {:?}", other.map(|p| p.3.len())),
        }
    }
}
