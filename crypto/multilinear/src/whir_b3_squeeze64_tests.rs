//! ★ The 64-byte-squeeze BLAKE3 configuration ([`Blake3Squeeze64Whir`])
//! against today's ([`Blake3Whir`]): the same trees, so the same roots; a
//! proof under either verifies under its own and is refused under the other,
//! because the two sponges draw different challenges from the same transcript.

use crypto::fiat_shamir::is_transcript::IsTranscript;
use crypto::merkle_tree::cap::CapPolicy;
use math::field::{
    element::FieldElement, extensions_goldilocks::Degree3GoldilocksExtensionField as Ext,
    goldilocks::GoldilocksField as F,
};

use crate::{
    mle::Mle,
    stacked_eval::{Claimed, StackedCommitment, StackedProof, prove, verify},
    stacking::{StackedLayout, borrow},
    whir::Domain,
    whir_chain::{
        ChainConfig, ChainFormat, FirstFold, GrindBits, GroupFold, Round0Trees, WhirFolds,
    },
    whir_commit::{Commitment, LeafOrder},
    whir_hash::{Blake3Squeeze64Whir, Blake3Whir, WhirHash},
};

type FE = FieldElement<F>;
type EE = FieldElement<Ext>;

const N: usize = 8;
const POLYS: usize = 3;

fn config() -> ChainConfig {
    ChainConfig {
        log_blowup: 2,
        log_folding: 4,
        num_queries: 5,
        grind: GrindBits::uniform(2),
        format: ChainFormat {
            folds: WhirFolds::First(FirstFold::new(6).unwrap()),
            group: GroupFold::Indexed,
            leaves: LeafOrder::Coset,
            arity4_cap: CapPolicy::Off,
            round0: Round0Trees::One,
            ..ChainFormat::DEFAULT
        },
    }
}

struct Fixture {
    columns: Vec<Mle<F>>,
    points: Vec<Vec<EE>>,
    values: Vec<EE>,
    layout: StackedLayout,
}

fn fixture() -> Fixture {
    let columns: Vec<Mle<F>> = (0..POLYS as u64)
        .map(|s| {
            Mle::new(
                (0..(1u64 << N))
                    .map(|i| {
                        FE::from(
                            i.wrapping_add(13 + 29 * s)
                                .wrapping_mul(0x9E37_79B9_7F4A_7C15)
                                >> 9,
                        )
                    })
                    .collect(),
            )
            .unwrap()
        })
        .collect();
    let points: Vec<Vec<EE>> = (0..POLYS as u64)
        .map(|i| {
            (0..N as u64)
                .map(|j| EE::from(5 + 17 * i + 3 * j))
                .collect()
        })
        .collect();
    let values = columns
        .iter()
        .zip(&points)
        .map(|(c, p)| c.evaluate_in(p).unwrap())
        .collect();
    Fixture {
        columns,
        points,
        values,
        layout: StackedLayout::build(&[N; POLYS], N).unwrap(),
    }
}

/// The proof of the fixture under `H`'s trees and sponge, its roots and its
/// domain.
fn prove_under<H: WhirHash>(f: &Fixture) -> (StackedProof<F, Ext>, Vec<Commitment>, Domain<F>) {
    let cfg = config();
    let stacked =
        StackedCommitment::<F, H>::commit(f.layout.clone(), &borrow(&f.columns), None, &cfg)
            .unwrap();
    let mut sponge = H::sponge();
    sponge.append_bytes(b"b3-squeeze64");
    let proof = prove::<F, Ext, _, H, _>(
        &stacked,
        &borrow(&f.columns),
        None,
        &Claimed::PerColumn(&f.points),
        &f.values,
        &cfg,
        &mut sponge,
    )
    .unwrap();
    (proof, stacked.roots(), stacked.domain().clone())
}

/// Whether `V`'s verifier, its sponge included, accepts `proof`.
fn accepts<V: WhirHash>(
    f: &Fixture,
    proof: &StackedProof<F, Ext>,
    roots: &[Commitment],
    domain: &Domain<F>,
) -> bool {
    let mut sponge = V::sponge();
    sponge.append_bytes(b"b3-squeeze64");
    verify::<F, Ext, _, V>(
        proof,
        &f.layout,
        roots,
        &Claimed::PerColumn(&f.points),
        &f.values,
        domain,
        &config(),
        &mut sponge,
    )
    .is_ok()
}

/// ★ Each configuration's proof verifies under its own verifier and is refused
/// by the other's; the trees (and so the roots) are the same, so what refuses
/// is the challenge stream alone.
#[test]
fn a_proof_under_one_squeeze_width_is_refused_under_the_other() {
    let f = fixture();
    let (narrow, narrow_roots, domain) = prove_under::<Blake3Whir>(&f);
    let (wide, wide_roots, _) = prove_under::<Blake3Squeeze64Whir>(&f);
    assert_eq!(narrow_roots, wide_roots, "the same trees");
    assert!(accepts::<Blake3Whir>(&f, &narrow, &narrow_roots, &domain));
    assert!(accepts::<Blake3Squeeze64Whir>(
        &f,
        &wide,
        &wide_roots,
        &domain
    ));
    assert!(
        !accepts::<Blake3Squeeze64Whir>(&f, &narrow, &narrow_roots, &domain),
        "a 32-byte-squeeze proof under the 64-byte verifier"
    );
    assert!(
        !accepts::<Blake3Whir>(&f, &wide, &wide_roots, &domain),
        "a 64-byte-squeeze proof under the 32-byte verifier"
    );
}
