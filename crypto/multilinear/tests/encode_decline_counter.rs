//! A successor encode the card declines is COUNTED (`gpu::encode_declines`),
//! and the fold the chain falls to proves the same bytes — for a chain of one
//! polynomial and for a grouped chain over a run of three
//! (`GroupFold::Indexed`). The encode switched off by its knob is the fold by
//! choice, not a decline, and counts nothing.
//!
//! ```text
//! cargo test --release -p multilinear --features cuda --test encode_decline_counter -- --test-threads=1
//! ```
//!
//! Needs a GPU. One `#[test]`: the arms switch process-wide overrides and read
//! process-wide counters, and a parallel test would move both.
#![cfg(feature = "cuda")]

use crypto::fiat_shamir::is_transcript::IsTranscript;
use math::field::element::FieldElement;
use math::field::extensions_goldilocks::Degree3GoldilocksExtensionField as Ext;
use math::field::goldilocks::GoldilocksField as F;
use multilinear::gpu;
use multilinear::mle::Mle;
use multilinear::stacked_eval::{self, Claimed, StackedCommitment};
use multilinear::stacking::StackedLayout;
use multilinear::whir_chain::{
    CapPolicy, ChainConfig, ChainFormat, FirstFold, GrindBits, GroupFold, WhirFolds,
};
use multilinear::whir_hash::{P1Whir, WhirHash};

type FE = FieldElement<F>;
type EE = FieldElement<Ext>;

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

fn column(num_vars: usize, seed: u64) -> Mle<F> {
    Mle::new(
        (0..(1u64 << num_vars))
            .map(|i| FE::from(i.wrapping_add(seed).wrapping_mul(0x9E37_79B9_7F4A_7C15) >> 11))
            .collect(),
    )
    .unwrap()
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Arm {
    Encoded,
    Declined,
    KnobOff,
}

/// Proves `polys` polynomials of `2^16` (one chain: per polynomial for one,
/// grouped for three) under `arm`; returns the bytes and what the encode and
/// decline counters moved by.
fn prove(polys: usize, arm: Arm) -> (Vec<u8>, u64, u64) {
    gpu::force_encode_successor(if arm == Arm::KnobOff {
        Some(false)
    } else {
        None
    });
    gpu::force_encode_decline(arm == Arm::Declined);
    let heights = vec![16usize; polys];
    let columns: Vec<Mle<F>> = (0..polys).map(|i| column(16, 5 + 17 * i as u64)).collect();
    let points: Vec<Vec<EE>> = (0..polys)
        .map(|i| {
            (0..16)
                .map(|j| EE::from(301 + 7 * i as u64 + j as u64))
                .collect()
        })
        .collect();
    let values: Vec<EE> = columns
        .iter()
        .zip(&points)
        .map(|(c, z)| c.evaluate_in(z).unwrap())
        .collect();
    let layout = StackedLayout::build(&heights, 16).unwrap();
    assert_eq!(layout.num_polys(), polys);
    let borrowed = multilinear::stacking::borrow(&columns);
    let sponge = || {
        let mut t = P1Whir::sponge();
        t.append_bytes(b"encode-decline");
        t
    };
    let (encoded, declined) = (gpu::encoded_successors(), gpu::encode_declines());
    let stacked =
        StackedCommitment::<F, P1Whir>::commit(layout.clone(), &borrowed, None, &config()).unwrap();
    let proof = stacked_eval::prove::<F, Ext, _, P1Whir, _>(
        &stacked,
        &borrowed,
        None,
        &Claimed::PerColumn(&points),
        &values,
        &config(),
        &mut sponge(),
    )
    .unwrap();
    let moved = (
        gpu::encoded_successors() - encoded,
        gpu::encode_declines() - declined,
    );
    stacked_eval::verify::<F, Ext, _, P1Whir>(
        &proof,
        &layout,
        &stacked.roots(),
        &Claimed::PerColumn(&points),
        &values,
        stacked.domain(),
        &config(),
        &mut sponge(),
    )
    .unwrap_or_else(|e| panic!("{polys} polynomials · {arm:?}: the verifier refused: {e:?}"));
    (
        rkyv::to_bytes::<rkyv::rancor::Error>(&proof)
            .unwrap()
            .to_vec(),
        moved.0,
        moved.1,
    )
}

#[test]
fn an_encode_the_card_declines_is_counted_and_proves_the_same_bytes() {
    for polys in [1usize, 3] {
        let (bytes, encoded, declined) = prove(polys, Arm::Encoded);
        assert_eq!(
            (encoded, declined),
            (1, 0),
            "{polys}: the one chain's successor encoded"
        );
        let (folded, encoded, declined) = prove(polys, Arm::Declined);
        assert_eq!(
            (encoded, declined),
            (0, 1),
            "{polys}: the declined encode counted, once, and nothing encoded"
        );
        assert!(
            folded == bytes,
            "{polys}: the fold proves the encode's bytes"
        );
        let (chosen, encoded, declined) = prove(polys, Arm::KnobOff);
        assert_eq!(
            (encoded, declined),
            (0, 0),
            "{polys}: the knob's fold is not a decline"
        );
        assert!(
            chosen == bytes,
            "{polys}: the knob's fold proves the same bytes"
        );
        println!(
            "ENCODE DECLINE {polys} polynomial(s): counted once, the fold the encode's {} bytes",
            bytes.len()
        );
    }
    gpu::force_encode_decline(false);
    gpu::force_encode_successor(None);
}
