//! Phase A's light local groups argued on the second card
//! (`LAMBDA_VM_LOCAL_LANE_SECOND_CARD`): a second lane uploads each light
//! local group to the two-card split's second card and runs its local bins
//! there, so the proof is the same byte for byte.
//!
//! Box tier, `#[ignore]` (a card, each run alone, deterministic grind): a
//! small block under local lookups proves the same bytes with the second
//! lane on as off, forced on one card (a second backend of card 0) and, on a
//! two-card box, on card 1; and with the second card's budget squeezed below
//! one group's store, every second-lane job is refused for memory and phase
//! B runs its bins, the same bytes. The lane's routing, stops, faults and
//! lines are host-tested in the stark crate (`local_lane_tests`), the knob
//! in `block_local_lookup_tests`.

#![cfg(feature = "cuda")]

use crate::block_whir::{
    self, BlockFormat, BlockStamps, BlockWhirProof, Deviations, LocalArgues,
    prove_block_whir_local_with,
};
use crate::tables::keccak_rnd::KeccakRndForm;
use crate::tests::block_whir_p1_tests::{many_groups, options};
use math_cuda::device::{
    TwoCardOpenings, TwoCardSplit, card_count, set_two_card_openings_for_test,
    set_two_card_split_for_test,
};
use multilinear::whir_chain::ArgueFormat;
use stark::multilinear_block::{LaneDefer, LaneState};
use stark::proof::options::{BaseFormat, ProofOptions};

/// Restores the process's two-card settings when dropped.
struct Restore;

impl Drop for Restore {
    fn drop(&mut self) {
        set_two_card_split_for_test(None);
        set_two_card_openings_for_test(None);
    }
}

fn format() -> BlockFormat {
    BlockFormat {
        argue: ArgueFormat::BATCHED,
        keccak_rnd: KeccakRndForm::LeanLocal,
        ..many_groups(BaseFormat::P1_WHIR)
    }
}

fn elf() -> Vec<u8> {
    crate::test_utils::asm_elf_bytes("test_keccak")
}

fn block_options() -> block_whir::BlockOptions {
    let mut o = options(0);
    o.window_log2 = Some(3);
    o
}

fn bytes(proof: &BlockWhirProof<LocalArgues>) -> Vec<u8> {
    rkyv::to_bytes::<rkyv::rancor::Error>(proof)
        .expect("serialize")
        .to_vec()
}

/// Host fallbacks, host widens, host fills and cross-card declines: the
/// second lane must move them as card 0's lane does, and the last not at all.
fn host_path_counters() -> [u64; 4] {
    [
        multilinear::gpu::host_fallbacks(),
        multilinear::narrow::host_widens().0,
        multilinear::records::fills().1,
        multilinear::gpu::cross_card_declines(),
    ]
}

/// The block proved with the second lane `on`, its bytes, its stamps and
/// how far it moved [`host_path_counters`]; it verifies.
fn proved(on: bool) -> (Vec<u8>, BlockStamps, [u64; 4]) {
    let elf = elf();
    let before = host_path_counters();
    let (proof, stamps) = prove_block_whir_local_with(
        &elf,
        &[],
        &ProofOptions::default_test_options(),
        &format(),
        &block_options(),
        &Deviations {
            local_lane: Some(true),
            local_lane_second: Some(on),
            ..Deviations::default()
        },
    )
    .expect("the block proves");
    let after = host_path_counters();
    eprintln!(
        "second lane {on}:\n{}",
        stark::multilinear_block::local_lane_lines(&stamps.groups)
    );
    let verified = crate::base_only::with_permit(|| {
        block_whir::verify_block_whir(
            &proof,
            &elf,
            &ProofOptions::default_test_options(),
            &format(),
        )
    })
    .unwrap_or(false);
    assert!(verified, "second lane {on}: the proof verifies");
    (
        bytes(&proof),
        stamps,
        std::array::from_fn(|k| after[k] - before[k]),
    )
}

fn needs_grind_and_card() {
    assert!(
        std::env::var_os("LAMBDA_VM_DETERMINISTIC_GRIND").is_some(),
        "the identity needs the deterministic grind"
    );
    assert!(multilinear::gpu::reserve_budget() > 0, "needs a card");
}

/// The local groups the second lane took: `(group, state)`.
fn on_second(stamps: &BlockStamps) -> Vec<(usize, LaneState)> {
    stamps
        .groups
        .iter()
        .enumerate()
        .filter(|(_, g)| g.zc_lane.local && g.zc_lane.card == 1)
        .map(|(i, g)| (i, g.zc_lane.state))
        .collect()
}

/// The lane off against on, under the split already set: the same bytes,
/// the same host paths, no cross-card read, and some local group run whole
/// on the second lane.
fn same_bytes_with_the_second_lane(what: &str) {
    let (off, off_stamps, off_counts) = proved(false);
    assert!(
        on_second(&off_stamps).is_empty(),
        "{what}: knob off, no job"
    );
    let (on, on_stamps, on_counts) = proved(true);
    let taken = on_second(&on_stamps);
    println!(
        "LOCAL LANE SECOND CARD {what}: groups on the second lane {taken:?} · host paths off \
         {off_counts:?} · on {on_counts:?}"
    );
    assert!(
        taken.iter().any(|(_, state)| *state == LaneState::Ran),
        "{what}: no local group ran on the second lane"
    );
    assert!(
        taken
            .iter()
            .all(|(_, state)| !matches!(state, LaneState::Declined | LaneState::Failed)),
        "{what}: {taken:?}"
    );
    assert_eq!(on, off, "{what}: the second lane moved a byte");
    assert_eq!(on_counts, off_counts, "{what}: another host path");
    assert_eq!(on_counts[3], 0, "{what}: a store read from the other card");
}

/// ★ C1. Card, one process: run alone, deterministic grind. Forced on one
/// card (a second backend of card 0, its own context, streams and ledger),
/// the second lane proves the same bytes as card 0's lane alone, with the
/// same host paths and no store read across backends (a shell still pointed
/// at card 0's store would be one).
#[test]
#[ignore = "card: run alone with LAMBDA_VM_DETERMINISTIC_GRIND=1"]
fn the_second_lane_proves_the_same_bytes_forced_on_one_card() {
    let _restore = Restore;
    needs_grind_and_card();
    set_two_card_openings_for_test(Some(TwoCardOpenings::Off));
    set_two_card_split_for_test(Some(TwoCardSplit::Forced));
    assert!(multilinear::gpu::two_card_split_ready(), "forced: ready");
    same_bytes_with_the_second_lane("forced on one card");
}

/// ★ C2. Two-card box, one process: run alone, deterministic grind. The
/// second lane on card 1 proves the same bytes, and no job of it was refused
/// for memory.
#[test]
#[ignore = "two cards: run alone with LAMBDA_VM_DETERMINISTIC_GRIND=1"]
fn the_second_lane_proves_the_same_bytes_on_card_1() {
    let _restore = Restore;
    needs_grind_and_card();
    assert!(card_count() >= 2, "needs two cards, saw {}", card_count());
    set_two_card_openings_for_test(Some(TwoCardOpenings::Off));
    set_two_card_split_for_test(Some(TwoCardSplit::Auto));
    assert!(multilinear::gpu::two_card_split_ready(), "card 1 ready");
    same_bytes_with_the_second_lane("on card 1");
    let (_, stamps, _) = proved(true);
    let memory: Vec<usize> = on_second(&stamps)
        .into_iter()
        .filter(|(_, state)| *state == LaneState::Deferred(LaneDefer::Memory))
        .map(|(g, _)| g)
        .collect();
    assert!(
        memory.is_empty(),
        "refused for memory on card 1: {memory:?}"
    );
}

/// ★ C4. Card, one process: run alone, deterministic grind. The second
/// card's budget squeezed below one group's store (and the lane's margins):
/// every second-lane job is refused for memory as it starts, phase B runs
/// its bins, and the bytes are card 0's lane's.
#[test]
#[ignore = "card: run alone with LAMBDA_VM_DETERMINISTIC_GRIND=1"]
fn a_squeezed_second_card_leaves_its_bins_to_phase_b() {
    let _restore = Restore;
    needs_grind_and_card();
    set_two_card_openings_for_test(Some(TwoCardOpenings::Off));
    set_two_card_split_for_test(Some(if card_count() >= 2 {
        TwoCardSplit::Auto
    } else {
        TwoCardSplit::Forced
    }));
    assert!(multilinear::gpu::two_card_split_ready(), "split ready");
    let (off, _, _) = proved(false);
    // 4 GiB: below the lane's two margins alone (2.5 GiB each), so no job
    // fits whatever its store.
    let was = multilinear::gpu::on_second_card(|| multilinear::gpu::set_reserve_budget(4 << 30));
    let squeezed = proved(true);
    multilinear::gpu::on_second_card(|| multilinear::gpu::set_reserve_budget(was));
    let (on, stamps, _) = squeezed;
    let taken = on_second(&stamps);
    assert!(!taken.is_empty(), "the second lane took no group");
    assert!(
        taken
            .iter()
            .all(|(_, state)| *state == LaneState::Deferred(LaneDefer::Memory)),
        "{taken:?}"
    );
    for (g, _) in &taken {
        assert!(
            !stamps.groups[*g].local_b.is_empty(),
            "group {g}: phase B ran its bins"
        );
    }
    assert_eq!(on, off, "a squeezed second card moved a byte");
}
