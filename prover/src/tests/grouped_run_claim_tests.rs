//! The column claims of one grouped run, from the prover crate (whose tests
//! CI runs): each column's batching power is indexed in the whole group, so a
//! claim cannot move between the run's polynomials with their sum kept. The
//! leaf replays the same batching (`lfm::whir_stacked`), and a leaf that
//! deviated would refuse honest proofs.

use crypto::fiat_shamir::is_transcript::IsTranscript;
use math::field::{
    element::FieldElement, extensions_goldilocks::Degree3GoldilocksExtensionField as Ext,
    goldilocks::GoldilocksField as F,
};
use multilinear::{
    mle::Mle,
    stacked_eval::{Claimed, StackedCommitment, prove, verify},
    stacking::{StackedLayout, borrow},
    whir_chain::{ChainConfig, ChainFormat, FirstFold, GrindBits, GroupFold, LeafOrder, WhirFolds},
    whir_hash::{P1Whir, WhirHash},
};

type FE = FieldElement<F>;
type EE = FieldElement<Ext>;

fn column(num_vars: usize, seed: u64) -> Mle<F> {
    Mle::new(
        (0..(1u64 << num_vars))
            .map(|i| FE::from(i.wrapping_add(seed).wrapping_mul(0x9E37_79B9_7F4A_7C15) >> 11))
            .collect(),
    )
    .unwrap()
}

fn sponge() -> <P1Whir as WhirHash>::Sponge {
    let mut t = P1Whir::sponge();
    t.append_bytes(b"a grouped run's claims");
    t
}

/// ★ A run of three polynomials (one column each) under the composed format
/// (one chain, coset order, blowup 2), proved over the TRUE polynomials for
/// claims that move one unit of value from polynomial 0's column to
/// polynomial 1's. The claims' sum is unchanged; only the column batching's
/// distinct exponents (each column's power indexed in the whole group) can
/// tell. Refused; the honest claims verify.
#[test]
fn a_claim_traded_between_a_runs_polynomials_is_refused() {
    let n = 8;
    let cfg = ChainConfig {
        log_blowup: 1,
        log_folding: 4,
        num_queries: 9,
        grind: GrindBits::default(),
        format: ChainFormat {
            folds: WhirFolds::First(FirstFold::new(6).unwrap()),
            group: GroupFold::Indexed,
            leaves: LeafOrder::Coset,
            ..ChainFormat::DEFAULT
        },
    };
    let columns: Vec<Mle<F>> = (0..3).map(|i| column(n, 5 + 31 * i)).collect();
    let points: Vec<Vec<EE>> = (0..3u64)
        .map(|i| {
            (0..n as u64)
                .map(|j| EE::from(7 + 11 * i + 3 * j))
                .collect()
        })
        .collect();
    let honest: Vec<EE> = columns
        .iter()
        .zip(&points)
        .map(|(c, p)| c.evaluate_in(p).unwrap())
        .collect();
    let layout = StackedLayout::build(&[n; 3], n).unwrap();
    assert_eq!(layout.num_polys(), 3);
    let stacked =
        StackedCommitment::<F, P1Whir>::commit(layout.clone(), &borrow(&columns), None, &cfg)
            .unwrap();
    let roots = stacked.roots();
    let run = |values: &[EE]| {
        let proof = prove::<F, Ext, _, P1Whir, _>(
            &stacked,
            &borrow(&columns),
            None,
            &Claimed::PerColumn(&points),
            values,
            &cfg,
            &mut sponge(),
        )
        .unwrap();
        assert_eq!(proof.polys.len(), 1, "one chain over the run of three");
        verify::<F, Ext, _, P1Whir>(
            &proof,
            &layout,
            &roots,
            &Claimed::PerColumn(&points),
            values,
            stacked.domain(),
            &cfg,
            &mut sponge(),
        )
    };
    assert_eq!(run(&honest), Ok(()), "the honest claims");
    let mut traded = honest.clone();
    traded[0] += EE::one();
    traded[1] = &traded[1] - EE::one();
    assert!(
        run(&traded).is_err(),
        "a claim traded between slots 0 and 1"
    );
}
