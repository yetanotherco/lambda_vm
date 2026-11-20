//! A stacked opening carrying its commit-time out-of-domain answers
//! (`multilinear::commit_ood`) with its round 0 on the card proves the SAME
//! bytes as the host alone, and the answers the card evaluates from resident
//! columns are the host's.
//!
//! ```text
//! cargo test --release -p multilinear --features cuda --test whir_commit_ood_device -- --test-threads=1
//! ```
//!
//! Needs a GPU. The card never writes an answer's `γ·eq(point, ·)` at the
//! stack's width: its share of each of the first rounds is added on the host
//! from the commit-time table, and what is left goes into the weight once the
//! table is spent or the first group of rounds ends
//! (`gpu::OpeningFactors::settle_overlay`). Cases: columns short enough that
//! the table covers the whole first fold (the production shape: written at
//! `2^(n − k0)`); columns tall enough that it runs out mid-fold (written
//! between two session rounds, and between a grouped part's rounds); and a
//! column filling the stack (no table: written at once, at full width). Each
//! one chain a polynomial and one a run (`GroupFold::PerPoly`, `Indexed`).
//! Arms: the host (the reference); the defaults; the share form; no lean
//! rounds; six; every successor folded; resident columns; and the host
//! fallback, the card's codewords with the factors declined. One `#[test]`:
//! the arms switch process-wide overrides, and parallel tests would race them.
#![cfg(feature = "cuda")]

use crypto::fiat_shamir::is_transcript::IsTranscript;
use math::field::element::FieldElement;
use math::field::extensions_goldilocks::Degree3GoldilocksExtensionField as Ext;
use math::field::goldilocks::GoldilocksField as F;
use multilinear::gpu;
use multilinear::mle::Mle;
use multilinear::stacked_eval::{
    self, Claimed, ColumnsAt, CommitOod, OodClaims, StackedCommitment,
};
use multilinear::stacking::StackedLayout;
use multilinear::whir_chain::{
    CapPolicy, ChainConfig, ChainFormat, FirstFold, GrindBits, GroupFold, WhirFolds,
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

/// First6 then uniform4, caps on every tree, no grind (a ground proof's nonce
/// is not reproducible across two searches).
fn config(group: GroupFold) -> ChainConfig {
    ChainConfig {
        log_blowup: 2,
        log_folding: 4,
        num_queries: 20,
        grind: GrindBits::default(),
        format: ChainFormat {
            cap: CapPolicy::Auto,
            folds: WhirFolds::First(FirstFold::new(6).unwrap()),
            group,
            ..ChainFormat::DEFAULT
        },
    }
}

struct Case {
    name: &'static str,
    layout: StackedLayout,
    columns: Vec<Mle<F>>,
    points: Vec<Vec<EE>>,
    values: Vec<EE>,
    z0: EE,
}

fn case_of(name: &'static str, heights: &[usize], polys: usize) -> Case {
    let columns: Vec<Mle<F>> = heights
        .iter()
        .enumerate()
        .map(|(i, &h)| column(h, 7 + 31 * i as u64))
        .collect();
    let layout = StackedLayout::build(heights, 16).unwrap();
    assert_eq!(layout.num_polys(), polys, "{name}: the stack's polynomials");
    let points: Vec<Vec<EE>> = heights
        .iter()
        .enumerate()
        .map(|(i, &h)| {
            (0..h)
                .map(|j| EE::from(101 + 13 * (i / 4) as u64 + j as u64))
                .collect()
        })
        .collect();
    let values = columns
        .iter()
        .zip(&points)
        .map(|(c, z)| c.evaluate_in(z).unwrap())
        .collect();
    Case {
        name,
        layout,
        columns,
        points,
        values,
        z0: EE::from(0x5eed_f00d_u64),
    }
}

fn cases() -> Vec<Case> {
    vec![
        // 2^10 columns: the table spans the whole first fold.
        case_of("short columns", &[10; 5 * 64], 5),
        // 2^13 columns: a three-variable table, spent mid-fold.
        case_of("tall columns", &[13; 5 * 8], 5),
        // A column filling the stack beside short ones: polynomial 0 has no
        // table at all.
        case_of("a full-height column", &[16, 10, 10, 10, 10], 2),
    ]
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Arm {
    Host,
    Defaults,
    ShareForm,
    Materialised,
    Lean(usize),
    Folded,
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

fn sponge<H: WhirHash>() -> H::Sponge {
    let mut t = H::sponge();
    t.append_bytes(b"whir-commit-ood-device");
    t
}

/// The case's claims, from the host's columns.
fn host_claims(c: &Case) -> OodClaims<Ext> {
    stacked_eval::ood_claims::<F, Ext, _>(
        &c.layout,
        &multilinear::stacking::borrow(&c.columns),
        None,
        &c.z0,
    )
    .unwrap()
    .0
}

/// One arm: commits, proves with the case's answers, has the host verifier
/// accept, and returns the proof's bytes and the roots.
fn prove_arm<H: WhirHash>(
    c: &Case,
    group: GroupFold,
    arm: Arm,
    resident: Option<&gpu::ResidentColumns>,
) -> (Vec<u8>, Vec<[u8; 32]>) {
    force(arm);
    let tag = format!("{} · {group:?} · {} · {arm:?}", c.name, H::NAME);
    let columns = multilinear::stacking::borrow(&c.columns);
    let claims = host_claims(c);
    let opens = gpu::open_calls();
    let stacked = StackedCommitment::<F, H>::commit(
        c.layout.clone(),
        &columns,
        resident.map(|s| (s, 0)),
        &config(group),
    )
    .unwrap();
    let proof = stacked_eval::prove_mapped_ood::<F, Ext, _, H, _>(
        &stacked,
        &columns,
        resident.map(|s| (s, ColumnsAt::From(0))),
        &Claimed::PerColumn(&c.points),
        &c.values,
        Some(&claims),
        &config(group),
        &mut sponge::<H>(),
    )
    .unwrap_or_else(|e| panic!("{tag}: the prover refused: {e:?}"));
    let opened = gpu::open_calls() - opens;
    match arm {
        Arm::Host => assert_eq!(opened, 0, "{tag}: an opening on the card"),
        // A declined opening's host-built tables may still go to the card.
        Arm::Fallback => {}
        _ => assert!(opened > 0, "{tag}: no opening on the card"),
    }
    stacked_eval::verify_ood::<F, Ext, _, H>(
        &proof,
        &c.layout,
        &stacked.roots(),
        &Claimed::PerColumn(&c.points),
        &c.values,
        Some(CommitOod {
            z0: &c.z0,
            answers: &claims.answers,
        }),
        stacked.domain(),
        &config(group),
        &mut sponge::<H>(),
    )
    .unwrap_or_else(|e| panic!("{tag}: the host verifier refused: {e:?}"));
    let bytes = rkyv::to_bytes::<rkyv::rancor::Error>(&proof)
        .unwrap()
        .to_vec();
    (bytes, stacked.roots())
}

fn every_arm<H: WhirHash>(c: &Case, group: GroupFold, store: &gpu::ResidentColumns) {
    let (host, host_roots) = prove_arm::<H>(c, group, Arm::Host, None);
    for arm in [
        Arm::Defaults,
        Arm::ShareForm,
        Arm::Materialised,
        Arm::Lean(6),
        Arm::Folded,
        Arm::Fallback,
    ] {
        let (card, roots) = prove_arm::<H>(c, group, arm, None);
        assert_eq!(roots, host_roots, "{} · {group:?} · {arm:?}: roots", c.name);
        assert!(
            card == host,
            "{} · {group:?} · {} · {arm:?}: the card must prove the host's bytes",
            c.name,
            H::NAME
        );
    }
    for arm in [Arm::Defaults, Arm::Lean(6)] {
        let (card, _) = prove_arm::<H>(c, group, arm, Some(store));
        assert!(
            card == host,
            "{} · {group:?} · {} · resident {arm:?}: the card must prove the host's bytes",
            c.name,
            H::NAME
        );
    }
    println!(
        "COMMIT OOD PARITY {} · {group:?} · {}: every arm the host's {} bytes",
        c.name,
        H::NAME,
        host.len()
    );
}

#[test]
fn an_opening_with_its_answers_on_the_card_proves_the_hosts_bytes() {
    for c in cases() {
        let store = gpu::upload_columns(&multilinear::stacking::borrow(&c.columns))
            .expect("the card took the columns (needs a GPU)");
        // The answers from resident columns, evaluated in place: the host's.
        release();
        let (card, counts) = stacked_eval::ood_claims::<F, Ext, _>(
            &c.layout,
            &multilinear::stacking::borrow(&c.columns),
            Some((&store, ColumnsAt::From(0))),
            &c.z0,
        )
        .unwrap();
        let host = host_claims(&c);
        assert_eq!(card.answers, host.answers, "{}: the answers", c.name);
        assert_eq!(card.columns, host.columns, "{}: the column values", c.name);
        assert!(
            counts.on_card > 0,
            "{}: no column evaluated on the card",
            c.name
        );
        println!(
            "COMMIT OOD ANSWERS {}: {} columns on the card, {} on the host",
            c.name, counts.on_card, counts.on_host
        );
        for group in [GroupFold::PerPoly, GroupFold::Indexed] {
            every_arm::<P1Whir>(&c, group, &store);
            every_arm::<RpxWhir>(&c, group, &store);
        }
    }
    release();
}
