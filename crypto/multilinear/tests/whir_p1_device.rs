//! D-WHIR-P1 S2: ZisK's Poseidon1 WHIR on the card. A chain whose codeword
//! stays on the card (commit, folds, every opening's 4-ary tree rebuilt there)
//! proves the SAME bytes as the chain over a host-held codeword, at every
//! arity-4 cap, odd and even tree depths; and an ext3 codeword's device tree is
//! the host's, paths and caps included.
//!
//! ```text
//! cargo test --release -p multilinear --features cuda --test whir_p1_device
//! ```
//!
//! Needs a GPU. The device path is asserted TAKEN, so a card that declined
//! could not turn this into a host-against-host comparison.
#![cfg(feature = "cuda")]

use crypto::fiat_shamir::is_transcript::IsTranscript;
use crypto::merkle_tree::cap::{CapPolicy, cap_len};
use math::field::element::FieldElement;
use math::field::extensions_goldilocks::Degree3GoldilocksExtensionField as Ext;
use math::field::goldilocks::GoldilocksField as F;
use multilinear::mle::Mle;
use multilinear::whir::{Domain, encode, lift_coefficients};
use multilinear::whir_chain::{
    ChainConfig, ChainFormat, GrindBits, RoundOpenings, commit, prove, verify,
};
use multilinear::whir_commit::{CodewordCommitment, tree_cap_height};
use multilinear::whir_hash::{P1Whir, WhirHash};

type FE = FieldElement<F>;
type EE = FieldElement<Ext>;

fn sponge() -> <P1Whir as WhirHash>::Sponge {
    let mut t = P1Whir::sponge();
    t.append_bytes(b"whir-p1-device");
    t
}

fn run(arity4_cap: CapPolicy, log_folding: usize) {
    // 2^16 evaluations at blowup 4: a 2^18 codeword, above the device commit
    // threshold, so the chain's first tree lives on the card.
    let num_vars = 16;
    let cfg = ChainConfig {
        log_blowup: 2,
        log_folding,
        num_queries: 25,
        grind: GrindBits::uniform(8),
        format: ChainFormat {
            arity4_cap,
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
    let tag = format!("k={log_folding} cap4={arity4_cap}");

    let (device, domain) = commit::<F, P1Whir>(&f, &cfg, true).unwrap();
    assert!(
        device.codeword().device().is_some(),
        "{tag}: the commit must have stayed on the card"
    );
    let device_proof =
        prove::<F, Ext, _, P1Whir>(&f, &z, &device, &domain, &cfg, &mut sponge()).unwrap();

    let host_domain = Domain::<F>::new(num_vars + cfg.log_blowup).unwrap();
    let host = CodewordCommitment::<F, P1Whir>::from_codeword_on_host(
        encode::<F, F>(&lift_coefficients(&f), &host_domain).unwrap(),
        cfg.schedule(num_vars)[0],
    )
    .unwrap();
    assert_eq!(host.root(), device.root(), "{tag}: roots");
    let host_proof =
        prove::<F, Ext, _, P1Whir>(&f, &z, &host, &host_domain, &cfg, &mut sponge()).unwrap();
    assert_eq!(
        rkyv::to_bytes::<rkyv::rancor::Error>(&device_proof)
            .unwrap()
            .as_slice(),
        rkyv::to_bytes::<rkyv::rancor::Error>(&host_proof)
            .unwrap()
            .as_slice(),
        "{tag}: the device chain must prove the host chain's bytes"
    );

    // Tree 0's paths: three siblings a 4-ary level below the cap, the owner
    // carrying the cap's real nodes.
    let caps = cfg.tree_caps_at(num_vars, 4);
    let depth0 = num_vars + cfg.log_blowup - cfg.schedule(num_vars)[0];
    assert_eq!(caps[0], tree_cap_height(arity4_cap, 25, depth0, 4));
    let walked = 3 * (depth0.div_ceil(2) - caps[0]);
    if let RoundOpenings::Base(p) = &device_proof.rounds[0].openings {
        let extra = if caps[0] > 0 {
            cap_len(depth0, caps[0], 4).unwrap()
        } else {
            0
        };
        assert_eq!(
            p.current[0].proof.merkle_path.len(),
            walked + extra,
            "{tag}"
        );
        assert_eq!(p.current[1].proof.merkle_path.len(), walked, "{tag}");
    } else {
        panic!("{tag}: round 0 opens base blocks");
    }

    verify::<F, Ext, _, P1Whir>(
        &device_proof,
        &device.root(),
        &z,
        y,
        &domain,
        &cfg,
        &mut sponge(),
    )
    .unwrap_or_else(|e| panic!("{tag}: the host verifier refused the device proof: {e:?}"));
}

/// Tree depths 14 (k = 4, even) and 15 (k = 3, odd: its top is two real nodes
/// and two padding digests), uncapped and at 4-ary caps 1 and 2.
#[test]
fn the_device_p1_chain_proves_the_host_chains_bytes() {
    for cap in [CapPolicy::Off, CapPolicy::Fixed(1), CapPolicy::Fixed(2)] {
        for k in [4usize, 3] {
            run(cap, k);
        }
    }
}

/// An ext3 codeword above the threshold: its device tree (the fold rounds'
/// `commit_tree_ext3`) is the host's — root, every queried path, and the
/// owner's cap — at odd and even depths.
#[test]
fn the_device_p1_ext3_tree_is_the_hosts() {
    for (log_len, k) in [(18usize, 4usize), (18, 3)] {
        let ext: Vec<EE> = (0..1u64 << log_len)
            .map(|i| {
                EE::new([
                    FE::from(i.wrapping_mul(0x9E37_79B9) >> 3),
                    FE::from(i),
                    FE::from(i ^ 0x55),
                ])
            })
            .collect();
        let before = multilinear::gpu::commit_calls();
        let device = CodewordCommitment::<Ext, P1Whir>::from_codeword(ext.clone(), k).unwrap();
        assert!(
            multilinear::gpu::commit_calls() > before,
            "log_len={log_len} k={k}: the tree must have been built on the card"
        );
        let host = CodewordCommitment::<Ext, P1Whir>::from_codeword_on_host(ext, k).unwrap();
        assert_eq!(device.root(), host.root(), "log_len={log_len} k={k}");
        let n = host.num_leaves();
        let indices: Vec<usize> = (0..20usize).map(|q| (q * 2_654_435_761) % n).collect();
        for c in [0usize, 1, 2] {
            assert_eq!(
                format!("{:?}", device.open_many_capped(&indices, c, true).unwrap()),
                format!("{:?}", host.open_many_capped(&indices, c, true).unwrap()),
                "log_len={log_len} k={k} c={c}"
            );
        }
    }
}
