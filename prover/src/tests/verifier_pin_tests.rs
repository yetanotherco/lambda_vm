//! What the block verifier holds as its own constants, from the prover crate
//! (whose tests CI runs): a WHIR chain's domain against the configured rate.

use crypto::fiat_shamir::default_transcript::DefaultTranscript;
use math::field::element::FieldElement;
use math::field::goldilocks::GoldilocksField as F;
use multilinear::mle::Mle;
use multilinear::whir::Domain;
use multilinear::whir_chain::{ChainConfig, ChainFormat, GrindBits, commit, prove, verify};
use multilinear::whir_hash::KeccakWhir;

type FE = FieldElement<F>;

/// ★ A WHIR chain verified against a domain off its configuration's rate is
/// refused by name (`Error::DomainRateMismatch`), one size smaller or one
/// larger; against its own domain it verifies.
#[test]
fn a_chain_on_a_domain_off_its_rate_is_refused() {
    let cfg = ChainConfig {
        log_blowup: 1,
        log_folding: 2,
        num_queries: 3,
        grind: GrindBits::default(),
        format: ChainFormat::DEFAULT,
    };
    let num_vars = 6;
    let f = Mle::new(
        (0..1u64 << num_vars)
            .map(|i| FE::from(i.wrapping_mul(0x9E37_79B9_7F4A_7C15) >> 13))
            .collect(),
    )
    .expect("a cube");
    let z: Vec<FE> = (0..num_vars).map(|i| FE::from(101 + i as u64)).collect();
    let y = f.evaluate(&z).expect("a point of the cube");
    let transcript = || DefaultTranscript::<F>::new(b"a chain's domain");
    let (commitment, domain) = commit::<F, KeccakWhir>(&f, &cfg, true).expect("commits");
    let proof = prove::<F, F, _, KeccakWhir>(&f, &z, &commitment, &domain, &cfg, &mut transcript())
        .expect("proves");
    let rate = num_vars + cfg.log_blowup;
    let check = |domain: &Domain<F>| {
        verify::<F, F, _, KeccakWhir>(
            &proof,
            &commitment.root(),
            &z,
            y,
            domain,
            &cfg,
            &mut transcript(),
        )
    };
    for log_size in [rate - 1, rate + 1] {
        assert_eq!(
            check(&Domain::<F>::new(log_size).expect("a domain")),
            Err(multilinear::Error::DomainRateMismatch {
                expected: rate,
                got: log_size
            }),
            "a domain of 2^{log_size} points"
        );
    }
    assert_eq!(check(&domain), Ok(()), "its own domain");
}
