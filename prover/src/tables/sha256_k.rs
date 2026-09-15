//! SHA256_K: verifier-committed 64-round constant table.
use super::{
    sha256_common::*,
    types::{BusId, FE, GoldilocksExtension, GoldilocksField},
};
use math::polynomial::Polynomial;
use stark::commitment::{ROWS_PER_LEAF, commit_bit_reversed};
use stark::prover::evaluate_polynomial_on_lde_domain;
use stark::{
    config::Commitment, lookup::BusInteraction, proof::options::ProofOptions, trace::TraceTable,
};
pub const WIDTH: usize = 3;
pub fn generate(n: usize) -> TraceTable<GoldilocksField, GoldilocksExtension> {
    trace(
        (0..64)
            .map(|i| vec![i as u64, executor::sha256::K[i] as u64, n as u64])
            .collect(),
        WIDTH,
    )
}
pub fn bus_interactions() -> Vec<BusInteraction> {
    vec![recv(BusId::ShaK, 2, vec![col(0), col(1)])]
}
pub fn preprocessed_commitment(options: &ProofOptions) -> Commitment {
    let columns: Vec<Vec<FE>> = vec![
        (0..64).map(|i| FE::from(i as u64)).collect(),
        executor::sha256::K
            .iter()
            .map(|k| FE::from(*k as u64))
            .collect(),
    ];
    let polys: Vec<_> = columns
        .iter()
        .map(|c| Polynomial::interpolate_fft::<GoldilocksField>(c).unwrap())
        .collect();
    let lde: Vec<_> = polys
        .iter()
        .map(|p| {
            evaluate_polynomial_on_lde_domain(
                p,
                options.blowup_factor as usize,
                64,
                &FE::from(options.coset_offset),
            )
            .unwrap()
        })
        .collect();
    commit_bit_reversed(&lde, ROWS_PER_LEAF).unwrap().1
}
