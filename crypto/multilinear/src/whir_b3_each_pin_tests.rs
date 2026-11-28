//! ★ The per-polynomial BLAKE3 round-0 format (`Round0Trees::Each`, one tree a
//! stacked polynomial), pinned: a fixed stack of five polynomials (a run of
//! four and one alone) committed and opened under 7-round BLAKE3 in both leaf
//! orders hashes to the digests below — the bytes this format had before one
//! tree a run existed under BLAKE3, so the rollback (`--trees each`) proves
//! the old format byte for byte.

use crypto::fiat_shamir::is_transcript::IsTranscript;
use crypto::hash::blake3::{BLAKE3_STANDARD_ROUNDS, Blake3Chain};
use crypto::merkle_tree::cap::CapPolicy;
use math::field::{
    element::FieldElement, extensions_goldilocks::Degree3GoldilocksExtensionField as Ext,
    goldilocks::GoldilocksField as F,
};

use crate::{
    mle::Mle,
    stacked_eval::{Claimed, StackedCommitment, prove},
    stacking::{StackedLayout, borrow},
    whir_chain::{
        ChainConfig, ChainFormat, FirstFold, GrindBits, GroupFold, Round0Trees, WhirFolds,
    },
    whir_commit::LeafOrder,
    whir_hash::{Blake3Whir, WhirHash},
};

type FE = FieldElement<F>;
type EE = FieldElement<Ext>;

fn digest_each(leaves: LeafOrder) -> String {
    const N: usize = 8;
    let cfg = ChainConfig {
        log_blowup: 2,
        log_folding: 4,
        num_queries: 3,
        grind: GrindBits::default(),
        format: ChainFormat {
            folds: WhirFolds::First(FirstFold::new(6).unwrap()),
            group: GroupFold::Indexed,
            leaves,
            arity4_cap: CapPolicy::Off,
            round0: Round0Trees::Each,
            ..ChainFormat::DEFAULT
        },
    };
    let polys = 5usize;
    let columns: Vec<Mle<F>> = (0..polys as u64)
        .map(|s| {
            Mle::new(
                (0..(1u64 << N))
                    .map(|i| {
                        FE::from(
                            i.wrapping_add(71 + 31 * s)
                                .wrapping_mul(0x9E37_79B9_7F4A_7C15)
                                >> 11,
                        )
                    })
                    .collect(),
            )
            .unwrap()
        })
        .collect();
    let points: Vec<Vec<EE>> = (0..polys as u64)
        .map(|i| {
            (0..N as u64)
                .map(|j| EE::from(7 + 11 * i + 3 * j))
                .collect()
        })
        .collect();
    let values: Vec<EE> = columns
        .iter()
        .zip(&points)
        .map(|(c, p)| c.evaluate_in(p).unwrap())
        .collect();
    let layout = StackedLayout::build(&vec![N; polys], N).unwrap();
    let stacked =
        StackedCommitment::<F, Blake3Whir>::commit(layout, &borrow(&columns), None, &cfg).unwrap();
    let mut sponge = Blake3Whir::sponge();
    sponge.append_bytes(b"b3-each-pin");
    let proof = prove::<F, Ext, _, Blake3Whir, _>(
        &stacked,
        &borrow(&columns),
        None,
        &Claimed::PerColumn(&points),
        &values,
        &cfg,
        &mut sponge,
    )
    .unwrap();
    let mut h = Blake3Chain::with_rounds(BLAKE3_STANDARD_ROUNDS);
    for root in stacked.roots() {
        h.update(&root);
    }
    h.update(
        rkyv::to_bytes::<rkyv::rancor::Error>(&proof)
            .unwrap()
            .as_slice(),
    );
    h.finalize_digest()
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect()
}

#[test]
fn the_per_tree_blake3_format_is_pinned() {
    let coset = digest_each(LeafOrder::Coset);
    let natural = digest_each(LeafOrder::Natural);
    println!("B3 EACH PIN coset {coset} natural {natural}");
    assert_eq!(
        coset,
        "0c01a974ba40792383b4f39542768f643a99ed4b36de8efe392b5c204d524898"
    );
    assert_eq!(
        natural,
        "53cf986cec11a2d984077f854a41d1e95115fd9a2fec9ac9552767ed674e1d6a"
    );
}
