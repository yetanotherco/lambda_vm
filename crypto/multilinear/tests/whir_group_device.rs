//! One WHIR chain per run of a group's stacked polynomials
//! (`GroupFold::Indexed`) with its round 0 on the card: the polynomials'
//! openings in lockstep, the index rounds over their bound tables laid slot by
//! slot, the successor encoded from the combined bound message (or each
//! codeword folded and the folds combined there) — proves the SAME bytes as
//! the host alone, codewords, folds, trees and factors all on the host.
//!
//! ```text
//! cargo test --release -p multilinear --features cuda --test whir_group_device -- --test-threads=1
//! ```
//!
//! Needs a GPU. Cases: runs of 2, 3 and 4; a group of 5 and one of 9 split into
//! runs [4, 1] and [4, 4, 1] (the runs of one are today's chains); a run of 3
//! whose columns' selectors straddle the index and whose polynomials take
//! different forms for their lean rounds (tile and share); and a 2^6 stack,
//! below every card threshold, where a card build proves on the host. Arms: the
//! defaults; the share form; the tile form checked against the share form on
//! every round; no lean rounds; four and six lean rounds; every successor
//! folded and combined on the card; resident columns; and the host fallback —
//! the card's codewords with the factors declined, which must be COUNTED.
//! Every path is asserted TAKEN in its arm by its counter, so a card that
//! declined could not turn a comparison into one of the host with itself. One
//! `#[test]`: the arms switch process-wide overrides, and parallel tests would
//! race them.
#![cfg(feature = "cuda")]

use crypto::fiat_shamir::is_transcript::IsTranscript;
use math::field::element::FieldElement;
use math::field::extensions_goldilocks::Degree3GoldilocksExtensionField as Ext;
use math::field::goldilocks::GoldilocksField as F;
use multilinear::gpu;
use multilinear::mle::Mle;
use multilinear::stacked_eval::{self, Claimed, StackedCommitment, index_chains};
use multilinear::stacking::StackedLayout;
use multilinear::whir_chain::{
    CapPolicy, ChainConfig, ChainFormat, FirstFold, GrindBits, GroupFold, RoundOpenings, WhirFolds,
};
use multilinear::whir_hash::{P1Whir, RpxWhir, WhirHash};

type FE = FieldElement<F>;
type EE = FieldElement<Ext>;

fn column(num_vars: usize, seed: u64) -> Mle<F> {
    Mle::new(
        (0..(1u64 << num_vars))
            .map(|i| {
                FE::from(
                    i.wrapping_add(seed)
                        .wrapping_mul(0x9E37_79B9_7F4A_7C15)
                        .wrapping_add(seed.wrapping_mul(6364136223846793005))
                        >> 11,
                )
            })
            .collect(),
    )
    .unwrap()
}

/// First6 then uniform4, caps on every tree, one chain per run of up to four
/// polynomials; no grind, whose nonce is not reproducible across two searches.
fn config() -> ChainConfig {
    ChainConfig {
        log_blowup: 2,
        log_folding: 4,
        num_queries: 20,
        grind: GrindBits::default(),
        format: ChainFormat {
            cap: CapPolicy::Auto,
            folds: WhirFolds::First(FirstFold::new(6).unwrap()),
            group: GroupFold::Indexed,
            ..ChainFormat::DEFAULT
        },
    }
}

/// Columns stacked as `heights` gives them, each claimed at the point of its
/// height (a table's columns share one), and what the run split derives.
struct Case {
    name: &'static str,
    layout: StackedLayout,
    columns: Vec<Mle<F>>,
    points: Vec<Vec<EE>>,
    values: Vec<EE>,
    /// Chains over two or more polynomials, the polynomials they open, and
    /// every chain.
    grouped: u64,
    grouped_polys: u64,
    chains: u64,
    /// Whether the stack is wide enough for the card to take it at all.
    on_card: bool,
}

fn case_of(name: &'static str, heights: &[usize], n_stack: usize, polys: usize) -> Case {
    let columns: Vec<Mle<F>> = heights
        .iter()
        .enumerate()
        .map(|(i, &h)| column(h, 7 + 31 * i as u64))
        .collect();
    let layout = StackedLayout::build(heights, n_stack).unwrap();
    assert_eq!(layout.num_polys(), polys, "{name}: the stack's polynomials");
    let points: Vec<Vec<EE>> = heights
        .iter()
        .map(|&h| {
            (0..h)
                .map(|j| EE::from(101 + 13 * h as u64 + j as u64))
                .collect()
        })
        .collect();
    let values = columns
        .iter()
        .zip(&points)
        .map(|(c, z)| c.evaluate_in(z).unwrap())
        .collect();
    let runs = index_chains(polys, GroupFold::Indexed);
    let grouped: Vec<_> = runs.iter().filter(|run| run.len() >= 2).collect();
    Case {
        name,
        layout,
        columns,
        points,
        values,
        grouped: grouped.len() as u64,
        grouped_polys: grouped.iter().map(|run| run.len() as u64).sum(),
        chains: runs.len() as u64,
        on_card: n_stack >= 14,
    }
}

fn cases() -> Vec<Case> {
    vec![
        case_of("run of 2", &[16, 15, 15], 16, 2),
        // Polynomial 2's columns run down to 2^3 rows: selectors of up to 13
        // bits, past the index after the first 6, and a stack the tile form
        // declines while polynomials 0 and 1 take it.
        case_of(
            "run of 3, mixed forms",
            &[
                15, 14, 13, 12, 12, 16, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 3,
            ],
            16,
            3,
        ),
        case_of("run of 4", &[16; 4], 16, 4),
        case_of("runs [4, 1]", &[16; 5], 16, 5),
        case_of("runs [4, 4, 1]", &[16; 9], 16, 9),
        case_of("below the card", &[6, 6, 5, 5], 6, 3),
    ]
}

/// Which path an arm proves on.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Arm {
    /// Every codeword, fold, tree and factor on the host: the reference.
    Host,
    /// Nothing forced: three lean rounds in tile form where a stack takes it,
    /// the fused fold, each first successor encoded from its bound message.
    Defaults,
    /// The lean rounds in share form (`LAMBDA_VM_WHIR_LEAN_TILES=0`).
    ShareForm,
    /// The tile form, every round and bound table compared with the share form
    /// (`=check`).
    TileCheck,
    /// No lean rounds: every opening written at full width up front.
    Materialised,
    /// Four, then six lean rounds (`LAMBDA_VM_WHIR_LEAN_ROUNDS`).
    Lean(usize),
    /// Every successor folded (`LAMBDA_VM_WHIR_ENCODE_SUCCESSOR=0`): a grouped
    /// one is each codeword folded and the folds combined on the card.
    Folded,
    /// The codewords on the card, every opening's factors declined there: the
    /// host's grouped round 0, which must be counted.
    Fallback,
}

fn force(arm: Arm) {
    gpu::force_host_whir_commit(arm == Arm::Host);
    gpu::force_shared_open_declined(matches!(arm, Arm::Host | Arm::Fallback));
    gpu::force_encode_successor(match arm {
        Arm::Folded => Some(false),
        _ => None,
    });
    gpu::force_lean_tiles(match arm {
        Arm::ShareForm => Some(gpu::LeanTiles::Off),
        Arm::TileCheck => Some(gpu::LeanTiles::Check),
        _ => None,
    });
    gpu::force_lean_rounds(match arm {
        Arm::Materialised => Some(0),
        Arm::Lean(rounds) => Some(rounds),
        _ => None,
    });
    gpu::force_fused_fold(None);
}

fn release() {
    gpu::force_host_whir_commit(false);
    gpu::force_shared_open_declined(false);
    gpu::force_encode_successor(None);
    gpu::force_lean_tiles(None);
    gpu::force_lean_rounds(None);
    gpu::force_fused_fold(None);
}

/// What the counters moved by over one proof.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct Moved {
    card_chains: u64,
    host_chains: u64,
    combined: u64,
    encoded: u64,
    opens: u64,
    lean: u64,
    tiles: u64,
    open_fallbacks: u64,
    commit_fallbacks: u64,
    rounds_compared: u64,
    rounds_differing: u64,
    positions_differing: u64,
}

fn counters() -> Moved {
    let (compared, differing) = gpu::lean_tile_checks();
    let (_, positions) = gpu::lean_tile_table_checks();
    Moved {
        card_chains: gpu::grouped_card_chains(),
        host_chains: gpu::grouped_host_chains(),
        combined: gpu::grouped_combined_successors(),
        encoded: gpu::encoded_successors(),
        opens: gpu::open_calls(),
        lean: gpu::lean_open_calls(),
        tiles: gpu::lean_tile_opens(),
        open_fallbacks: gpu::open_host_fallbacks(),
        commit_fallbacks: gpu::host_fallbacks(),
        rounds_compared: compared,
        rounds_differing: differing,
        positions_differing: positions,
    }
}

fn since(before: Moved) -> Moved {
    let now = counters();
    Moved {
        card_chains: now.card_chains - before.card_chains,
        host_chains: now.host_chains - before.host_chains,
        combined: now.combined - before.combined,
        encoded: now.encoded - before.encoded,
        opens: now.opens - before.opens,
        lean: now.lean - before.lean,
        tiles: now.tiles - before.tiles,
        open_fallbacks: now.open_fallbacks - before.open_fallbacks,
        commit_fallbacks: now.commit_fallbacks - before.commit_fallbacks,
        rounds_compared: now.rounds_compared - before.rounds_compared,
        rounds_differing: now.rounds_differing - before.rounds_differing,
        positions_differing: now.positions_differing - before.positions_differing,
    }
}

fn sponge<H: WhirHash>() -> H::Sponge {
    let mut t = H::sponge();
    t.append_bytes(b"whir-group-device");
    t
}

/// One arm: commits, proves, has the host verifier accept, checks the arm's
/// counters, and returns the proof's bytes and the roots.
fn prove_arm<H: WhirHash>(
    c: &Case,
    arm: Arm,
    resident: Option<&gpu::ResidentColumns>,
) -> (Vec<u8>, Vec<[u8; 32]>) {
    force(arm);
    let tag = format!("{} · {} · {arm:?}", c.name, H::NAME);
    let columns = multilinear::stacking::borrow(&c.columns);
    let before = counters();
    let stacked = StackedCommitment::<F, H>::commit(
        c.layout.clone(),
        &columns,
        resident.map(|s| (s, 0)),
        &config(),
    )
    .unwrap();
    let proof = stacked_eval::prove::<F, Ext, _, H, _>(
        &stacked,
        &columns,
        resident.map(|s| (s, 0)),
        &Claimed::PerColumn(&c.points),
        &c.values,
        &config(),
        &mut sponge::<H>(),
    )
    .unwrap_or_else(|e| panic!("{tag}: the prover refused: {e:?}"));
    let moved = since(before);
    stacked_eval::verify::<F, Ext, _, H>(
        &proof,
        &c.layout,
        &stacked.roots(),
        &Claimed::PerColumn(&c.points),
        &c.values,
        stacked.domain(),
        &config(),
        &mut sponge::<H>(),
    )
    .unwrap_or_else(|e| panic!("{tag}: the host verifier refused: {e:?}"));

    // The shape the split derives: one chain a run, the grouped variant at the
    // grouped chains' round 0.
    assert_eq!(proof.polys.len() as u64, c.chains, "{tag}: chains");
    let grouped = proof
        .polys
        .iter()
        .filter(|chain| matches!(chain.rounds[0].openings, RoundOpenings::BaseGrouped(_)))
        .count() as u64;
    assert_eq!(grouped, c.grouped, "{tag}: grouped chains");

    let card = c.on_card;
    match arm {
        Arm::Host => {
            assert_eq!(moved.card_chains, 0, "{tag}: a grouped round 0 on the card");
            assert_eq!(
                moved.host_chains, 0,
                "{tag}: the host path counted as a fallback"
            );
            assert_eq!(moved.opens, 0, "{tag}: an opening on the card");
            assert_eq!(moved.combined, 0, "{tag}: a combine on the card");
            assert_eq!(moved.encoded, 0, "{tag}: an encode on the card");
        }
        Arm::Fallback if card => {
            assert_eq!(moved.card_chains, 0, "{tag}: a grouped round 0 on the card");
            assert_eq!(
                moved.host_chains, c.grouped,
                "{tag}: every grouped round 0 counted as a host fallback"
            );
            assert_eq!(
                moved.open_fallbacks,
                c.grouped_polys + (c.chains - c.grouped),
                "{tag}: every polynomial's factors counted on the host"
            );
            assert_eq!(
                moved.combined, c.grouped,
                "{tag}: every grouped successor folded and combined on the card"
            );
            // A run of one falls back as today: its host-built tables go to
            // the card whole, and its successor is encoded from them there.
            assert_eq!(
                moved.encoded,
                c.chains - c.grouped,
                "{tag}: successors encoded (the runs of one alone)"
            );
        }
        _ if card => {
            assert_eq!(
                moved.card_chains, c.grouped,
                "{tag}: every grouped round 0 on the card"
            );
            assert_eq!(moved.host_chains, 0, "{tag}: grouped host fallbacks");
            assert_eq!(moved.open_fallbacks, 0, "{tag}: openings on the host");
            assert_eq!(moved.commit_fallbacks, 0, "{tag}: commits on the host");
            assert_eq!(
                moved.opens,
                c.grouped_polys + (c.chains - c.grouped),
                "{tag}: one card opening a polynomial"
            );
            let (encoded, combined) = if arm == Arm::Folded {
                (0, c.grouped)
            } else {
                (c.chains, 0)
            };
            assert_eq!(moved.encoded, encoded, "{tag}: successors encoded");
            assert_eq!(moved.combined, combined, "{tag}: successors combined");
            let lean = if arm == Arm::Materialised {
                0
            } else {
                moved.opens
            };
            assert_eq!(moved.lean, lean, "{tag}: lean openings");
            if arm == Arm::ShareForm || arm == Arm::Materialised {
                assert_eq!(moved.tiles, 0, "{tag}: an opening in tile form");
            }
            if arm == Arm::TileCheck {
                assert!(moved.tiles > 0, "{tag}: no opening in tile form");
                assert!(moved.rounds_compared > 0, "{tag}: no round compared");
            }
            assert_eq!(moved.rounds_differing, 0, "{tag}: tile rounds differing");
            assert_eq!(moved.positions_differing, 0, "{tag}: tile tables differing");
        }
        // Below the card's thresholds every arm is the host's path, and none
        // of it is a fallback.
        _ => {
            assert_eq!(moved.card_chains, 0, "{tag}: a grouped round 0 on the card");
            assert_eq!(
                moved.host_chains, 0,
                "{tag}: the host path counted as a fallback"
            );
        }
    }
    let bytes = rkyv::to_bytes::<rkyv::rancor::Error>(&proof)
        .unwrap()
        .to_vec();
    (bytes, stacked.roots())
}

fn every_arm<H: WhirHash>(c: &Case) {
    let (host, host_roots) = prove_arm::<H>(c, Arm::Host, None);
    for arm in [
        Arm::Defaults,
        Arm::ShareForm,
        Arm::TileCheck,
        Arm::Materialised,
        Arm::Lean(4),
        Arm::Lean(6),
        Arm::Folded,
        Arm::Fallback,
    ] {
        let (card, card_roots) = prove_arm::<H>(c, arm, None);
        assert_eq!(
            card_roots,
            host_roots,
            "{} · {} · {arm:?}: roots",
            c.name,
            H::NAME
        );
        assert!(
            card == host,
            "{} · {} · {arm:?}: the card must prove the host's bytes",
            c.name,
            H::NAME
        );
    }
    if c.on_card {
        let store = gpu::upload_columns(&multilinear::stacking::borrow(&c.columns))
            .expect("the card took the columns (needs a GPU)");
        for arm in [Arm::Defaults, Arm::Lean(6), Arm::Folded] {
            let (card, card_roots) = prove_arm::<H>(c, arm, Some(&store));
            assert_eq!(
                card_roots,
                host_roots,
                "{} · {} · resident {arm:?}: roots",
                c.name,
                H::NAME
            );
            assert!(
                card == host,
                "{} · {} · resident {arm:?}: the card must prove the host's bytes",
                c.name,
                H::NAME
            );
        }
    }
    println!(
        "GROUPED PARITY {} · {}: {} chains ({} grouped over {} polynomials) · every arm the host's {} bytes",
        c.name,
        H::NAME,
        c.chains,
        c.grouped,
        c.grouped_polys,
        host.len()
    );
}

#[test]
fn a_grouped_chain_on_the_card_proves_the_hosts_bytes() {
    let cases = cases();
    // Which of the mixed run's polynomials the tile form takes at three lean
    // rounds: the first two, not the third.
    let mixed = &cases[1];
    let takes: Vec<bool> = (0..mixed.layout.num_polys())
        .map(|poly| {
            let raw: Vec<(usize, Vec<u64>, [u64; 3])> = mixed
                .points
                .iter()
                .zip(mixed.layout.placements())
                .filter(|(_, place)| place.poly == poly)
                .map(|(point, place)| {
                    let limbs = point
                        .iter()
                        .flat_map(|z| gpu::ext3_raw(z).unwrap())
                        .collect();
                    (place.offset, limbs, [1, 0, 0])
                })
                .collect();
            multilinear::lean_tiles::plan(&raw, 16, 3).is_some()
        })
        .collect();
    assert_eq!(
        takes,
        [true, true, false],
        "which polynomials the tile form takes"
    );
    for c in &cases {
        every_arm::<P1Whir>(c);
        every_arm::<RpxWhir>(c);
    }
    release();
}
