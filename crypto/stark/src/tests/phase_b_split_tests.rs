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

/// A failed opening, and a panicking opening thread, on either driver are the
/// prove's error (never the other driver's stop, never a panic).
#[test]
fn a_failed_or_panicking_opening_on_either_driver_is_the_proves_error() {
    let n = SIZES.len();
    for place in 0..n {
        let (failed, _) = prove_split(
            ArgueFormat::PerTable,
            PhaseBOverlap::On,
            &in_order(),
            true,
            Split::OnHost,
            &vec![false; n],
            OverlapFaults {
                open_fails: Some(place),
                ..OverlapFaults::default()
            },
        );
        assert!(failed.is_err(), "place {place}: a failed opening proved");
        if let Err(MlError::DeviceFailed { stage }) = &failed {
            assert_ne!(*stage, "the other card's phase B failed", "place {place}");
        }
        let (panicked, _) = prove_split(
            ArgueFormat::PerTable,
            PhaseBOverlap::On,
            &in_order(),
            true,
            Split::OnHost,
            &vec![false; n],
            OverlapFaults {
                open_panics: Some(place),
                ..OverlapFaults::default()
            },
        );
        // The last group taken on each driver opens alone, on its own
        // thread: no opener to panic there.
        if let Err(e) = &panicked {
            assert!(
                matches!(e, MlError::DeviceFailed { stage } if *stage != "the other card's phase B failed"),
                "place {place}: {e:?}"
            );
        }
    }
}

/// The re-ranking hook under two drivers: each driver's turns call it, and
/// the proof is the one-driver proof with the same hook.
#[test]
fn a_re_ranking_hook_under_two_drivers_proves_the_same_bytes() {
    use crate::multilinear_block::PhaseBReorder;
    let n = SIZES.len();
    // At the second turn, the rest reversed.
    let hook = |p: usize, order: &[usize]| -> Option<Vec<usize>> {
        (p == 1).then(|| order[p + 1..].iter().rev().copied().collect())
    };
    let hook: &PhaseBReorder<'_> = &hook;
    for argue in ARGUES {
        let prove_with = |split: Split| {
            let keep = vec![false; n];
            let set = move |block: &mut BlockCommitted<'_, Fp, Ext>| {
                if split == Split::OnHost {
                    block.set_two_card_split(Some(keep.clone()), true);
                }
            };
            prove_block_with::<H>(
                &config(argue),
                PhaseBOverlap::On,
                &in_order(),
                OverlapFaults::default(),
                &set,
                Some(hook),
                true,
                None,
            )
            .0
            .expect("proves")
        };
        let one = prove_with(Split::None);
        let two = prove_with(Split::OnHost);
        assert!(cards(&two).contains(&1), "{argue:?}: nothing split");
        assert_eq!(
            bytes(&two),
            bytes(&one),
            "{argue:?}: the hook under two drivers moved a byte"
        );
        verify(argue, &two).expect("verifies");
    }
}

/// Both drivers fail (each at a place of its own, the failure set to two
/// places): the prove's error is a failure's own, never the other driver's
/// stop.
#[test]
fn of_two_failures_the_answer_is_a_failure_not_a_stop() {
    let n = SIZES.len();
    let (proved, _) = prove_split(
        ArgueFormat::PerTable,
        PhaseBOverlap::Off,
        &in_order(),
        true,
        Split::OnHost,
        &vec![false; n],
        OverlapFaults {
            argue_fails: Some(1),
            open_fails: Some(0),
            ..OverlapFaults::default()
        },
    );
    match proved {
        Err(MlError::DeviceFailed { stage }) => {
            assert_ne!(stage, "the other card's phase B failed")
        }
        Err(other) => panic!("{other:?}"),
        Ok(_) => panic!("proved with two failures"),
    }
}

/// Run separately with LAMBDA_VM_CARDS set to one, two or four ordinals.
#[cfg(feature = "cuda")]
#[test]
#[ignore = "card: run alone with an explicit card list"]
fn indexed_cards_assemble_shuffled_phase_b() {
    let topology = multilinear::gpu::indexed_cards()
        .expect("configured backends initialize")
        .expect("set LAMBDA_VM_CARDS");
    let ordinals: Vec<usize> = std::env::var("LAMBDA_VM_CARDS")
        .expect("explicit list")
        .split(',')
        .map(|ordinal| ordinal.trim().parse().expect("CUDA ordinal"))
        .collect();
    for (&card, ordinal) in topology.iter().zip(ordinals) {
        assert_eq!(
            multilinear::gpu::card_memory(card)
                .expect("backend memory")
                .0,
            ordinal
        );
    }
    let argue = ArgueFormat::PerTable;
    let keep = [false; SIZES.len()];
    let (serial, baseline) = prove_split(
        argue,
        PhaseBOverlap::Off,
        &in_order(),
        true,
        Split::None,
        &keep,
        OverlapFaults::default(),
    );
    let reversed: Vec<_> = in_order().into_iter().rev().collect();
    let declines = multilinear::gpu::cross_card_declines();
    let (shuffled, pieces) = prove_split(
        argue,
        PhaseBOverlap::On,
        &reversed,
        true,
        Split::OptedIn,
        &keep,
        OverlapFaults::default(),
    );
    let shuffled = shuffled.expect("indexed drivers prove");
    let mut ran: Vec<_> = cards(&shuffled).into_iter().map(usize::from).collect();
    ran.sort_unstable();
    ran.dedup();
    assert_eq!(ran, topology);
    assert_eq!(multilinear::gpu::cross_card_declines(), declines);
    assert_ne!(
        pieces.iter().map(|(g, _)| *g).collect::<Vec<_>>(),
        in_order()
    );
    assert_eq!(sorted(pieces), sorted(baseline));
    assert_eq!(
        bytes(&shuffled),
        bytes(&serial.expect("serial reference proves"))
    );
    verify(argue, &shuffled).expect("verifies");
}

#[cfg(feature = "cuda")]
#[test]
#[ignore = "card: two explicit backends, openings and retention off"]
fn indexed_secondary_admission_fallback_preserves_proof_bytes() {
    use multilinear::gpu;

    let topology = gpu::indexed_cards().unwrap().expect("set LAMBDA_VM_CARDS");
    assert_eq!(topology, [0, 1]);
    let prove = || {
        let (result, _) = prove_split(
            ArgueFormat::PerTable,
            PhaseBOverlap::Off,
            &in_order(),
            true,
            Split::OptedIn,
            &[false; SIZES.len()],
            OverlapFaults::default(),
        );
        let proof = result.expect("both drivers prove");
        let used = cards(&proof);
        assert!(used.contains(&0) && used.contains(&1));
        verify(ArgueFormat::PerTable, &proof).expect("verifies");
        proof
    };
    let declines = gpu::cross_card_declines();
    let baseline = prove();
    let refusals = gpu::room_revive_refusals();
    let _tight = gpu::override_card_budget(1, 1, None).expect("secondary has a ledger");
    let fallback = prove();
    assert!(
        gpu::room_revive_refusals() > refusals,
        "revive room refused"
    );
    assert_eq!(gpu::cross_card_declines(), declines);
    assert_eq!(bytes(&fallback), bytes(&baseline));
    let archives = |proof: &BlockProved<Fp, Ext>| {
        use rkyv::{rancor::Error, to_bytes};
        [
            to_bytes::<Error>(&proof.0).unwrap(),
            to_bytes::<Error>(&proof.1).unwrap(),
            to_bytes::<Error>(&proof.2).unwrap(),
            to_bytes::<Error>(&proof.4).unwrap(),
            to_bytes::<Error>(&proof.5).unwrap(),
        ]
    };
    assert!(
        archives(&fallback)
            .iter()
            .zip(&archives(&baseline))
            .all(|(fallback, baseline)| fallback.as_slice() == baseline.as_slice()),
        "secondary admission fallback changed the raw archives"
    );
}

/// Run with four aliases and retention disabled, in its own process because
/// this test intentionally quarantines a card.
#[cfg(feature = "cuda")]
#[test]
#[ignore = "card: run alone with four aliases"]
fn indexed_driver_error_joins_and_releases_its_work() {
    use multilinear::gpu;
    let topology = gpu::indexed_cards().unwrap().expect("set LAMBDA_VM_CARDS");
    assert_eq!(topology.len(), 4);
    let held = || {
        topology
            .iter()
            .map(|&card| gpu::in_indexed_card(card, gpu::ledger_reserved))
            .collect::<Vec<_>>()
    };
    let before = held();
    let (result, _) = prove_split(
        ArgueFormat::PerTable,
        PhaseBOverlap::On,
        &in_order(),
        true,
        Split::OptedIn,
        &[false; SIZES.len()],
        OverlapFaults {
            argue_fails: Some(0),
            ..OverlapFaults::default()
        },
    );
    assert!(matches!(
        result,
        Err(MlError::DeviceFailed {
            stage: "a test's failed argue"
        })
    ));
    assert_eq!(held(), before, "every driver's promises have been returned");
    assert_eq!(gpu::indexed_cards().unwrap().unwrap().len(), 3);
}

#[cfg(feature = "cuda")]
#[test]
#[ignore = "card: two explicit backends, openings auto, retention off"]
fn migrated_opening_error_quarantines_its_backend_and_allows_a_fresh_prove() {
    use math_cuda::device;
    use multilinear::gpu;

    let topology = gpu::indexed_cards().unwrap().expect("set LAMBDA_VM_CARDS");
    assert_eq!(topology, [0, 1]);
    assert!(
        gpu::two_card_openings(),
        "set LAMBDA_VM_TWO_CARD_OPENINGS=auto"
    );
    let prove = |faults| {
        prove_split(
            ArgueFormat::PerTable,
            PhaseBOverlap::On,
            &in_order(),
            true,
            Split::OptedIn,
            &[true, true, true, true, true, true, false],
            faults,
        )
        .0
    };
    let settled = || {
        topology
            .iter()
            .map(|&card| {
                gpu::in_indexed_card(card, || {
                    device::drain_untrimmed().unwrap();
                    (gpu::ledger_reserved(), device::pool_used_bytes().unwrap().0)
                })
            })
            .collect::<Vec<_>>()
    };
    let warm = prove(OverlapFaults::default()).expect("warm proof");
    assert_eq!(cards(&warm), [0, 0, 0, 0, 0, 0, 1]);
    let baseline = settled();
    let failed = prove(OverlapFaults {
        migrated_open_fails: Some(0),
        ..OverlapFaults::default()
    });
    assert!(matches!(
        failed,
        Err(MlError::DeviceFailed {
            stage: "a test's migrated opening after upload",
        })
    ));
    assert_eq!(gpu::indexed_cards().unwrap().unwrap(), [0]);
    assert_eq!(settled(), baseline);
    let fresh = prove(OverlapFaults::default()).expect("fresh proof with card 0 pins");
    assert_eq!(cards(&fresh), [0; SIZES.len()]);
    verify(ArgueFormat::PerTable, &fresh).expect("fresh proof verifies");
    assert_eq!(bytes(&fresh), bytes(&warm));
}

#[cfg(feature = "cuda")]
#[test]
#[ignore = "card: run alone with four aliases"]
fn indexed_commit_error_drops_peer_receipts() {
    use multilinear::gpu;
    let cards = gpu::indexed_cards().unwrap().expect("set LAMBDA_VM_CARDS");
    assert_eq!(cards.len(), 4);
    let held = || {
        cards
            .iter()
            .map(|&card| gpu::in_indexed_card(card, gpu::ledger_reserved))
            .collect::<Vec<_>>()
    };
    let before = held();
    let all_started = std::sync::Barrier::new(cards.len());
    let result = crate::phase_b_split::commit_indexed(
        vec![(); cards.len()],
        &cards,
        &[0, 1, 2, 3],
        |_, g, ()| {
            let room = gpu::reserve_room(1 << 20);
            all_started.wait();
            if g == 1 {
                return Err("a test's failed commit");
            }
            room.ok_or("card refused the test reservation")
        },
        |_| "a committer panicked",
    );
    assert!(matches!(result, Err("a test's failed commit")));
    assert_eq!(held(), before, "every lane's receipt has been released");
}

#[test]
fn inproc_driver_reverses_phase_b_and_releases_work_on_error() {
    use crate::multilinear_block::InprocWorkers;
    use multilinear::gpu;
    use rkyv::{rancor::Error, to_bytes};

    let workers = InprocWorkers::parse("inproc:4").unwrap().unwrap();
    let order: Vec<_> = in_order().into_iter().rev().collect();
    let config = config(ArgueFormat::PerTable);
    let prove = |remote, faults| {
        prove_block_with::<H>(
            &config,
            PhaseBOverlap::Off,
            &order,
            faults,
            &|block| {
                if remote {
                    block.set_inproc(workers, Vec::new(), vec![Vec::new(); SIZES.iter().sum()]);
                }
            },
            None,
            true,
            None,
        )
    };
    let before = gpu::ledger_reserved();
    let (failed, observed) = prove(
        true,
        OverlapFaults {
            // At least one earlier group completed and sent a GroupDone.
            open_fails: Some(2),
            ..OverlapFaults::default()
        },
    );
    assert!(matches!(failed, Err(MlError::DeviceFailed { .. })));
    assert!(observed.iter().all(|(g, _)| order[..2].contains(g)));
    if gpu::reserve_budget() > before && !gpu::columns_disabled() {
        let peak = gpu::reserved_window_peak();
        assert!(peak > before, "the failed turn had a live CUDA reservation");
        eprintln!("Cluster cancellation CUDA reservation peaked at {peak} bytes");
    }
    assert_eq!(
        gpu::ledger_reserved(),
        before,
        "the failed attempt released its leases"
    );

    let (local, observed_local) = prove(false, OverlapFaults::default());
    let (remote, observed_remote) = prove(true, OverlapFaults::default());
    let local = local.unwrap();
    let remote = remote.unwrap();
    assert_eq!(sorted(observed_local), sorted(observed_remote));
    assert_eq!(
        to_bytes::<Error>(&remote.0).unwrap().as_slice(),
        to_bytes::<Error>(&local.0).unwrap().as_slice()
    );
    assert_eq!(
        to_bytes::<Error>(&remote.1).unwrap().as_slice(),
        to_bytes::<Error>(&local.1).unwrap().as_slice()
    );
    assert_eq!(
        to_bytes::<Error>(&remote.2).unwrap().as_slice(),
        to_bytes::<Error>(&local.2).unwrap().as_slice()
    );
    assert_eq!(
        to_bytes::<Error>(&remote.4).unwrap().as_slice(),
        to_bytes::<Error>(&local.4).unwrap().as_slice()
    );
    verify(ArgueFormat::PerTable, &remote).unwrap();
    assert_eq!(
        gpu::ledger_reserved(),
        before,
        "the fresh attempt released its leases"
    );
}

fn prove_with_worker_fault(
    group: usize,
    fault: crate::multilinear_block::WorkerFault,
) -> (Result<BlockProved<Fp, Ext>, MlError>, Vec<Heard>) {
    use crate::multilinear_block::InprocWorkers;

    let workers = InprocWorkers::parse("inproc:4")
        .unwrap()
        .unwrap()
        .with_fault(group, fault);
    let order: Vec<_> = std::iter::once(group)
        .chain(in_order().into_iter().filter(|&g| g != group))
        .collect();
    prove_block_with::<H>(
        &config(ArgueFormat::PerTable),
        PhaseBOverlap::Off,
        &order,
        OverlapFaults::default(),
        &|block| block.set_inproc(workers, Vec::new(), vec![Vec::new(); SIZES.iter().sum()]),
        None,
        true,
        None,
    )
}

#[test]
fn inproc_rejects_invalid_group_contents_before_observing_the_group() {
    use crate::multilinear_block::WorkerFault::*;

    for fault in [SuccessorRoot, OodValue, Sumcheck, OpeningPath] {
        let (result, heard) = prove_with_worker_fault(0, fault);
        assert!(
            matches!(result, Err(MlError::ColumnOpeningRejected { .. })),
            "{fault:?} did not reach the host opening verifier: {:?}",
            result.err(),
        );
        assert!(heard.iter().all(|(g, _)| *g != 0), "{fault:?}");
    }
}

#[test]
fn inproc_rejects_short_and_long_chains_including_prepared() {
    use crate::multilinear_block::WorkerFault::*;

    for (group, fault) in [
        (0, ExtraRound),
        (0, MissingRound),
        (1, MissingPreparedRound),
    ] {
        let (result, heard) = prove_with_worker_fault(group, fault);
        assert!(
            matches!(result, Err(MlError::ColumnOpeningRejected { .. })),
            "{fault:?}: {:?}",
            result.err(),
        );
        assert!(heard.iter().all(|(g, _)| *g != group), "{fault:?}");
    }
}

#[test]
fn inproc_phase_b_worker_failure_preserves_the_error_and_releases_work() {
    use crate::multilinear_block::WorkerFault;
    use multilinear::gpu;

    let before = gpu::ledger_reserved();
    let (failed, heard) = prove_with_worker_fault(0, WorkerFault::Fail);
    assert!(matches!(
        failed,
        Err(MlError::ClusterProtocol("a test's failed phase B worker"))
    ));
    assert!(heard.is_empty());
    if gpu::reserve_budget() > before && !gpu::columns_disabled() {
        let peak = gpu::reserved_window_peak();
        assert!(peak > before, "phase B held a CUDA reservation");
        eprintln!("Worker failure CUDA reservation peaked at {peak} bytes");
    }
    assert_eq!(gpu::ledger_reserved(), before);

    let (proof, heard) = prove_split(
        ArgueFormat::PerTable,
        PhaseBOverlap::Off,
        &in_order(),
        true,
        Split::None,
        &[],
        OverlapFaults::default(),
    );
    assert_eq!(heard.len(), SIZES.len());
    verify(ArgueFormat::PerTable, &proof.unwrap()).unwrap();
    assert_eq!(gpu::ledger_reserved(), before);
}
