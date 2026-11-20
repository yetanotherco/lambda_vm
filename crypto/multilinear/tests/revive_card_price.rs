//! A retired stack's planned revive prices (`RetiredStack::revive_over_store_card_bytes`
//! and `revive_whole_card_bytes`), with coset openings forced on: over a
//! store on coset leaves the revive holds its room and no codeword, whole it
//! holds every codeword besides; on natural leaves the two hold the same
//! codewords. Host only. It forces a process-wide knob, so it is the only
//! test in its binary.

use math::field::element::FieldElement;
use math::field::goldilocks::GoldilocksField as F;
use multilinear::gpu;
use multilinear::mle::Mle;
use multilinear::stacked_eval::StackedCommitment;
use multilinear::stacking::{StackedLayout, borrow};
use multilinear::whir_chain::{
    ChainConfig, ChainFormat, FirstFold, GrindBits, LeafOrder, WhirFolds,
};
use multilinear::whir_commit::TreeDrop;
use multilinear::whir_hash::P1Whir;

fn config(leaves: LeafOrder) -> ChainConfig {
    ChainConfig {
        log_blowup: 2,
        log_folding: 4,
        num_queries: 33,
        grind: GrindBits::default(),
        format: ChainFormat {
            folds: WhirFolds::First(FirstFold::new(6).unwrap()),
            leaves,
            ..ChainFormat::DEFAULT
        },
    }
}

fn column(num_vars: usize, seed: u64) -> Mle<F> {
    Mle::new(
        (0..(1u64 << num_vars))
            .map(|i| FieldElement::<F>::from(i.wrapping_mul(0x9E37_79B9).wrapping_add(seed) >> 3))
            .collect(),
    )
    .unwrap()
}

#[test]
fn a_revive_over_a_store_prices_no_codeword_on_coset_leaves() {
    gpu::force_coset_openings(Some(true));
    let result = std::panic::catch_unwind(|| {
        for (leaves, coset) in [(LeafOrder::Coset, true), (LeafOrder::Natural, false)] {
            let config = config(leaves);
            let columns: Vec<Mle<F>> = (0..3).map(|k| column(10, k)).collect();
            let layout = StackedLayout::build(&[10, 10, 10], 11).unwrap();
            let polys = layout.num_polys() as u64;
            let codeword = (1u64 << (layout.n_stack() + config.log_blowup)) * 8;
            let retired =
                StackedCommitment::<F, P1Whir>::commit_on_host(layout, &borrow(&columns), &config)
                    .unwrap()
                    .retire(TreeDrop::uniform(0), &config)
                    .unwrap();
            let over = retired.revive_over_store_card_bytes(&config);
            let whole = retired.revive_whole_card_bytes(&config);
            if coset {
                assert!(
                    whole >= over + polys * codeword,
                    "{leaves:?}: over {over}, whole {whole}"
                );
                assert!(
                    over < codeword,
                    "{leaves:?}: a revive over the store holds no codeword ({over})"
                );
            } else {
                assert!(
                    over >= polys * codeword,
                    "{leaves:?}: natural leaves re-encode ({over})"
                );
                assert!(whole >= polys * codeword, "{leaves:?}: {whole}");
            }
        }
    });
    gpu::force_coset_openings(None);
    if let Err(e) = result {
        std::panic::resume_unwind(e);
    }
}
