//! W1 end to end on the device: a WHIR chain whose codeword stays on the card
//! (commit, folds, and every opening's tree rebuilt there) proves, under the
//! `Auto` cap, the SAME bytes as the chain over a host-held codeword, and the
//! host verifier accepts it.
//!
//! ```text
//! cargo test --release -p multilinear --features cuda --test whir_cap_device
//! ```
//!
//! Needs a GPU. The device path is asserted TAKEN (the first commitment's
//! codeword is on the card), so a card that declined could not turn this into
//! a host-against-host comparison.
#![cfg(feature = "cuda")]

use crypto::fiat_shamir::default_transcript::DefaultTranscript;
use math::field::element::FieldElement;
use math::field::extensions_goldilocks::Degree3GoldilocksExtensionField as Ext;
use math::field::goldilocks::GoldilocksField as F;
use multilinear::mle::Mle;
use multilinear::whir::{Domain, encode, lift_coefficients};
use multilinear::whir_chain::{
    CapPolicy, ChainConfig, ChainFormat, GrindBits, RoundOpenings, commit, prove, verify,
};
use multilinear::whir_commit::CodewordCommitment;
use multilinear::whir_hash::{KeccakWhir, RpxWhir, WhirHash};

type FE = FieldElement<F>;
type EE = FieldElement<Ext>;

fn run<H: WhirHash>(cap: CapPolicy) {
    // 2^16 evaluations at blowup 4: a 2^18 codeword, above the device commit
    // threshold, so the chain's first tree lives on the card.
    let num_vars = 16;
    let cfg = ChainConfig {
        log_blowup: 2,
        log_folding: 4,
        num_queries: 25,
        grind: GrindBits::default(),
        format: ChainFormat {
            cap,
            ..ChainFormat::DEFAULT
        },
    };
    let f = Mle::new(
        (0..(1u64 << num_vars))
            .map(|i| FE::from(i.wrapping_mul(6364136223846793005).wrapping_add(17) >> 11))
            .collect(),
    )
    .unwrap();
    let z: Vec<EE> = (0..num_vars).map(|i| EE::from(301 + i as u64)).collect();
    let y = f.evaluate_in(&z).unwrap();
    let tag = format!("{} cap={cap}", H::NAME);

    let (device, domain) = commit::<F, H>(&f, &cfg, true).unwrap();
    assert!(
        device.codeword().device().is_some(),
        "{tag}: the commit must have stayed on the card, or this compares the host with itself"
    );
    let device_proof = prove::<F, Ext, _, H>(
        &f,
        &z,
        &device,
        &domain,
        &cfg,
        &mut DefaultTranscript::<Ext>::new(b"whir-cap-device"),
    )
    .unwrap();

    let host_domain = Domain::<F>::new(num_vars + cfg.log_blowup).unwrap();
    let host = CodewordCommitment::<F, H>::from_codeword_on_host(
        encode::<F, F>(&lift_coefficients(&f), &host_domain).unwrap(),
        cfg.schedule(num_vars)[0],
    )
    .unwrap();
    assert_eq!(host.root(), device.root(), "{tag}: roots");
    let host_proof = prove::<F, Ext, _, H>(
        &f,
        &z,
        &host,
        &host_domain,
        &cfg,
        &mut DefaultTranscript::<Ext>::new(b"whir-cap-device"),
    )
    .unwrap();

    let a = rkyv::to_bytes::<rkyv::rancor::Error>(&device_proof).unwrap();
    let b = rkyv::to_bytes::<rkyv::rancor::Error>(&host_proof).unwrap();
    assert_eq!(
        a.as_slice(),
        b.as_slice(),
        "{tag}: the device chain must prove the host chain's bytes"
    );

    // The owner path of tree 0 carries its cap.
    let caps = cfg.tree_caps(num_vars);
    let depth0 = num_vars + cfg.log_blowup - cfg.schedule(num_vars)[0];
    if let RoundOpenings::Base(p) = &device_proof.rounds[0].openings {
        let extra = if caps[0] > 0 { 1usize << caps[0] } else { 0 };
        assert_eq!(
            p.current[0].proof.merkle_path.len(),
            depth0 - caps[0] + extra
        );
        assert_eq!(p.current[1].proof.merkle_path.len(), depth0 - caps[0]);
    } else {
        panic!("{tag}: round 0 opens base blocks");
    }

    verify::<F, Ext, _, H>(
        &device_proof,
        &device.root(),
        &z,
        y,
        &domain,
        &cfg,
        &mut DefaultTranscript::<Ext>::new(b"whir-cap-device"),
    )
    .unwrap_or_else(|e| panic!("{tag}: the host verifier refused the device proof: {e:?}"));
}

#[test]
fn the_device_chain_proves_the_host_chains_bytes_under_the_cap() {
    for cap in [CapPolicy::Off, CapPolicy::Auto, CapPolicy::Fixed(5)] {
        run::<KeccakWhir>(cap);
        run::<RpxWhir>(cap);
    }
}
