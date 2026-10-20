//! The two memory kernels end to end on the device: a stacked opening whose
//! first rounds run over its shares (the lean opening) and whose codeword
//! folds run every level in one launch proves the SAME bytes as today's
//! device path (factors materialised up front, one fold launch per level), and
//! the host verifier accepts both.
//!
//! ```text
//! cargo test --release -p multilinear --features cuda --test whir_lean_device
//! ```
//!
//! Four arms: the path before the kernels (lean rounds 0, folds level by
//! level), six lean rounds with the fused fold (`LAMBDA_VM_WHIR_LEAN_ROUNDS=6`,
//! the way back), three lean rounds and the fused fold with every successor
//! folded (`LAMBDA_VM_WHIR_ENCODE_SUCCESSOR=0`, the way back), and the
//! defaults (three lean rounds, the fused fold, the first successor encoded
//! from the bound message instead of folded). Needs a GPU. Each new path is
//! asserted TAKEN in its arms and NOT in the others (the call counters), so a
//! card that declined could not turn this into a comparison of today's path
//! with itself. One `#[test]`: the arms switch process-wide overrides, and
//! parallel tests would race them.
//!
//! Three more arms run the lean rounds in TILE form
//! (`LAMBDA_VM_WHIR_LEAN_TILES=1`, `=wide`, `=check`) on a second stack, one
//! the form takes: the same bytes again, every opening counted in tile form,
//! and under `check` every round and every opening's bound tables compared
//! with the share form on the card with none different. On the first stack — a column of eight rows, which the
//! form declines — the same arms prove today's bytes in share form.
#![cfg(feature = "cuda")]

use crypto::fiat_shamir::default_transcript::DefaultTranscript;
use math::field::element::FieldElement;
use math::field::extensions_goldilocks::Degree3GoldilocksExtensionField as Ext;
use math::field::goldilocks::GoldilocksField as F;
use multilinear::gpu;
use multilinear::mle::Mle;
use multilinear::stacked_eval::{self, Claimed, StackedCommitment, StackedProof};
use multilinear::stacking::StackedLayout;
use multilinear::whir_chain::{
    CapPolicy, ChainConfig, ChainFormat, FirstFold, GrindBits, WhirFolds,
};
use multilinear::whir_hash::{KeccakWhir, RpxWhir, WhirHash};

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

/// First6 then uniform4, caps on every tree, no grind: a ground proof's nonce
/// is not reproducible across two searches, and the grind is not what moved.
fn config() -> ChainConfig {
    ChainConfig {
        log_blowup: 2,
        log_folding: 4,
        num_queries: 20,
        grind: GrindBits::default(),
        format: ChainFormat {
            cap: CapPolicy::Auto,
            folds: WhirFolds::First(FirstFold::new(6).unwrap()),
            ..ChainFormat::DEFAULT
        },
    }
}

/// Columns of mixed heights in a `2^16` stack — above the device commit and
/// sumcheck thresholds — each claimed at its own point.
struct Case {
    layout: StackedLayout,
    columns: Vec<Mle<F>>,
    points: Vec<Vec<EE>>,
    values: Vec<EE>,
}

fn case() -> Case {
    let heights = [13usize, 13, 13, 12, 12, 10, 8, 6, 3];
    case_of(&heights, &[0, 1, 2, 3, 4, 5, 6, 7, 8])
}

/// A stack the tile form takes: three tables of several columns each (their
/// columns share a point), no column under `2^9` rows.
fn tile_case() -> Case {
    let heights = [13usize, 13, 13, 12, 12, 12, 11, 9, 9];
    case_of(&heights, &[0, 0, 0, 1, 1, 1, 2, 3, 3])
}

/// Columns of `heights` in one stack of `2^16`, column `i` claimed at the
/// point of table `tables[i]`.
fn case_of(heights: &[usize], tables: &[usize]) -> Case {
    let columns: Vec<Mle<F>> = heights
        .iter()
        .enumerate()
        .map(|(i, &h)| column(h, 7 + 31 * i as u64))
        .collect();
    let layout = StackedLayout::build(heights, 16).unwrap();
    assert_eq!(layout.num_polys(), 1);
    let points: Vec<Vec<EE>> = heights
        .iter()
        .zip(tables)
        .map(|(&h, &table)| {
            (0..h)
                .map(|j| EE::from(101 + 13 * table as u64 + j as u64))
                .collect()
        })
        .collect();
    let values = columns
        .iter()
        .zip(&points)
        .map(|(c, z)| c.evaluate_in(z).unwrap())
        .collect();
    Case {
        layout,
        columns,
        points,
        values,
    }
}

/// Which path an arm proves on.
#[derive(Clone, Copy, Debug)]
enum Arm {
    /// No lean rounds, every fold level by level: the path before the kernels.
    Materialised,
    /// Six lean rounds and the fused fold (`LAMBDA_VM_WHIR_LEAN_ROUNDS=6`).
    Six,
    /// Three lean rounds and the fused fold, every successor folded
    /// (`LAMBDA_VM_WHIR_ENCODE_SUCCESSOR=0`).
    Folded,
    /// Nothing forced: the defaults, three lean rounds, the fused fold and
    /// each chain's first successor encoded from its bound message.
    Defaults,
    /// The defaults with the lean rounds in tile form
    /// (`LAMBDA_VM_WHIR_LEAN_TILES`): on, wide, or checked against the share
    /// form on every round.
    Tiles(gpu::LeanTiles),
}

/// One arm: the path before the kernels or the two kernels.
///
/// `tiled` says whether the stack is one the tile form takes: its arms then
/// open every chain in tile form, and in share form otherwise.
fn prove_arm<H: WhirHash>(
    c: &Case,
    arm: Arm,
    resident: Option<&gpu::ResidentColumns>,
    tiled: bool,
) -> (StackedProof<F, Ext>, Vec<[u8; 32]>) {
    let lean = !matches!(arm, Arm::Materialised);
    gpu::force_encode_successor(match arm {
        Arm::Defaults | Arm::Tiles(_) => None,
        _ => Some(false),
    });
    gpu::force_lean_tiles(Some(match arm {
        Arm::Tiles(tiles) => tiles,
        _ => gpu::LeanTiles::Off,
    }));
    match arm {
        Arm::Materialised => {
            gpu::force_lean_rounds(Some(0));
            gpu::force_fused_fold(Some(false));
        }
        Arm::Six => {
            gpu::force_lean_rounds(Some(6));
            gpu::force_fused_fold(Some(true));
        }
        Arm::Folded | Arm::Defaults | Arm::Tiles(_) => {
            gpu::force_lean_rounds(None);
            gpu::force_fused_fold(None);
            if std::env::var_os("LAMBDA_VM_WHIR_LEAN_ROUNDS").is_none() {
                assert_eq!(
                    gpu::lean_rounds_in_effect(),
                    3,
                    "the default is three lean rounds"
                );
            }
            if matches!(arm, Arm::Defaults | Arm::Tiles(_))
                && std::env::var_os("LAMBDA_VM_WHIR_ENCODE_SUCCESSOR").is_none()
            {
                assert!(
                    gpu::encode_successor_in_effect(),
                    "the default encodes the first successor"
                );
            }
        }
    }
    let columns = multilinear::stacking::borrow(&c.columns);
    let stacked = StackedCommitment::<F, H>::commit(
        c.layout.clone(),
        &columns,
        resident.map(|store| (store, 0)),
        &config(),
    )
    .unwrap();
    let (lean_before, fused_before) = (gpu::lean_open_calls(), gpu::fused_fold_calls());
    let encoded_before = gpu::encoded_successors();
    let (opens_before, folds_before) = (gpu::open_calls(), gpu::resident_fold_calls());
    let (tiles_before, checks_before) = (gpu::lean_tile_opens(), gpu::lean_tile_checks());
    let tables_before = gpu::lean_tile_table_checks();
    let proof = stacked_eval::prove::<F, Ext, _, H, _>(
        &stacked,
        &columns,
        resident.map(|store| (store, 0)),
        &Claimed::PerColumn(&c.points),
        &c.values,
        &config(),
        &mut DefaultTranscript::<Ext, H::Transcript>::new(b"whir-lean-device"),
    )
    .unwrap();
    let (lean_opens, fused_folds) = (
        gpu::lean_open_calls() - lean_before,
        gpu::fused_fold_calls() - fused_before,
    );
    // The totals the production log reads each share against: both arms open
    // and fold on the device, so a share of zero is the path off, never a
    // path that did not run.
    let (opens, folds) = (
        gpu::open_calls() - opens_before,
        gpu::resident_fold_calls() - folds_before,
    );
    assert!(opens > 0, "{}: no opening ran on the device", H::NAME);
    assert!(folds > 0, "{}: no fold ran on the device", H::NAME);
    // Each chain's first successor, and only in its arm: one a chain, and the
    // chain's later folds still fold.
    assert_eq!(
        gpu::encoded_successors() - encoded_before,
        if matches!(arm, Arm::Defaults | Arm::Tiles(_)) && gpu::encode_successor_in_effect() {
            opens
        } else {
            0
        },
        "{} {arm:?}: successors encoded",
        H::NAME
    );
    // The tile form in its arms on a stack that takes it, every chain; never
    // elsewhere. Under `check`, each chain's lean rounds compared with the
    // share form, none different.
    let tile_opens = gpu::lean_tile_opens() - tiles_before;
    let (checked, mismatched) = gpu::lean_tile_checks();
    let checked = (checked - checks_before.0, mismatched - checks_before.1);
    let (tables, positions) = gpu::lean_tile_table_checks();
    let tables = (tables - tables_before.0, positions - tables_before.1);
    match arm {
        Arm::Tiles(tiles) if tiled => {
            assert_eq!(
                tile_opens,
                opens,
                "{} {arm:?}: every opening in tile form",
                H::NAME
            );
            let (rounds, written) = if tiles == gpu::LeanTiles::Check {
                (opens * gpu::lean_rounds_in_effect() as u64, opens)
            } else {
                (0, 0)
            };
            assert_eq!(
                checked,
                (rounds, 0),
                "{} {arm:?}: rounds compared, differing",
                H::NAME
            );
            assert_eq!(
                tables,
                (written, 0),
                "{} {arm:?}: bound tables compared, positions differing",
                H::NAME
            );
        }
        _ => {
            assert_eq!(
                tile_opens,
                0,
                "{} {arm:?}: an opening in tile form",
                H::NAME
            );
            assert_eq!(checked, (0, 0), "{} {arm:?}: rounds compared", H::NAME);
            assert_eq!(tables, (0, 0), "{} {arm:?}: tables compared", H::NAME);
        }
    }
    if lean {
        assert!(
            lean_opens > 0,
            "{}: the lean opening was not taken",
            H::NAME
        );
        assert!(fused_folds > 0, "{}: the fused fold was not taken", H::NAME);
        assert_eq!(
            (lean_opens, fused_folds),
            (opens, folds),
            "{}: every opening lean and every fold fused",
            H::NAME
        );
    } else {
        assert_eq!(
            lean_opens,
            0,
            "{}: today's arm took the lean opening",
            H::NAME
        );
        assert_eq!(
            fused_folds,
            0,
            "{}: today's arm took the fused fold",
            H::NAME
        );
    }
    stacked_eval::verify::<F, Ext, _, H>(
        &proof,
        &c.layout,
        &stacked.roots(),
        &Claimed::PerColumn(&c.points),
        &c.values,
        stacked.domain(),
        &config(),
        &mut DefaultTranscript::<Ext, H::Transcript>::new(b"whir-lean-device"),
    )
    .unwrap_or_else(|e| panic!("{}: the host verifier refused the proof: {e:?}", H::NAME));
    (proof, stacked.roots())
}

fn both_arms<H: WhirHash>(c: &Case, resident: Option<&gpu::ResidentColumns>, tiled: bool) {
    let (today, today_roots) = prove_arm::<H>(c, Arm::Materialised, resident, tiled);
    let today_bytes = rkyv::to_bytes::<rkyv::rancor::Error>(&today).unwrap();
    for arm in [
        Arm::Six,
        Arm::Folded,
        Arm::Defaults,
        Arm::Tiles(gpu::LeanTiles::On),
        Arm::Tiles(gpu::LeanTiles::Wide),
        Arm::Tiles(gpu::LeanTiles::Check),
    ] {
        let (lean, lean_roots) = prove_arm::<H>(c, arm, resident, tiled);
        assert_eq!(today_roots, lean_roots, "{} {arm:?}: roots", H::NAME);
        assert_eq!(
            rkyv::to_bytes::<rkyv::rancor::Error>(&lean)
                .unwrap()
                .as_slice(),
            today_bytes.as_slice(),
            "{} ({}, {arm:?}): the kernels must prove today's bytes",
            H::NAME,
            if resident.is_some() {
                "resident"
            } else {
                "parts"
            },
        );
    }
}

#[test]
fn the_memory_kernels_prove_todays_bytes() {
    // The first stack has a column of eight rows, which the tile form
    // declines; the second is one it takes.
    for (c, tiled) in [(case(), false), (tile_case(), true)] {
        let raw: Vec<(usize, Vec<u64>, [u64; 3])> = c
            .points
            .iter()
            .zip(c.layout.placements())
            .map(|(point, place)| {
                let limbs = point
                    .iter()
                    .flat_map(|z| gpu::ext3_raw(z).unwrap())
                    .collect();
                (place.offset, limbs, [1, 0, 0])
            })
            .collect();
        assert_eq!(
            multilinear::lean_tiles::plan(&raw, 16, 3).is_some(),
            tiled,
            "which stack the tile form takes"
        );
        both_arms::<RpxWhir>(&c, None, tiled);
        both_arms::<KeccakWhir>(&c, None, tiled);
        let store = gpu::upload_columns(&multilinear::stacking::borrow(&c.columns))
            .expect("the card took the columns (needs a GPU)");
        both_arms::<RpxWhir>(&c, Some(&store), tiled);
    }
    gpu::force_lean_rounds(None);
    gpu::force_fused_fold(None);
    gpu::force_encode_successor(None);
    gpu::force_lean_tiles(None);
}
