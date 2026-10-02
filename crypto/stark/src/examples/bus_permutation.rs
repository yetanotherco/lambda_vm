//! A self-balancing LogUp table built on [`AirWithBuses`] — the AIR type every
//! production VM table is — for tests that must prove on the DEVICE.
//!
//! The CUDA composition arm evaluates constraints from the AIR's captured IR
//! (`AIR::constraint_program`). `AirWithBuses` supplies it; the hand-written
//! example AIRs (`LogReadOnlyRAP`, `FibonacciRAP`, …) do not, and the trait's
//! default panics by design. A test whose trace crosses the GPU LDE threshold
//! therefore needs an AIR like this one, whatever it is testing.
//!
//! Layout: four main columns `a, b, c, d` with `b` a permutation of `a` and `d`
//! a permutation of `c`; four interactions (`a` sent and `b` received on one
//! bus, `c` sent and `d` received on another), so the aux trace has one
//! committed term column and the accumulated column. The table balances on its
//! own: its bus contribution is zero, which is what a single-table verify
//! expects.

use math::field::element::FieldElement;
use math::field::extensions_goldilocks::Degree3GoldilocksExtensionField;
use math::field::goldilocks::GoldilocksField;

use crate::constraints::builder::EmptyConstraints;
use crate::lookup::{
    AirWithBuses, AuxiliaryTraceBuildData, BusInteraction, Multiplicity,
    NullBoundaryConstraintBuilder, Packing,
};
use crate::proof::options::ProofOptions;
use crate::trace::TraceTable;

type F = GoldilocksField;
type E = Degree3GoldilocksExtensionField;
type FE = FieldElement<F>;

/// The AIR: no table constraints of its own, the LogUp ones from the framework.
pub type BusPermutationAir =
    AirWithBuses<F, E, NullBoundaryConstraintBuilder, (), EmptyConstraints>;

const BUS_AB: u64 = 1;
const BUS_CD: u64 = 2;

/// The AIR under `options`.
pub fn bus_permutation_air(options: &ProofOptions) -> BusPermutationAir {
    let one = |col: usize| Packing::Direct.columns(&[col]);
    AirWithBuses::new(
        4,
        AuxiliaryTraceBuildData {
            interactions: vec![
                BusInteraction::sender(BUS_AB, Multiplicity::One, one(0)),
                BusInteraction::receiver(BUS_AB, Multiplicity::One, one(1)),
                BusInteraction::sender(BUS_CD, Multiplicity::One, one(2)),
                BusInteraction::receiver(BUS_CD, Multiplicity::One, one(3)),
            ],
        },
        options,
        1,
        EmptyConstraints,
    )
}

/// A `rows`-row trace (`rows` a power of two, at least 2): `b` is `a`
/// reversed, `d` is `c` rotated by one row.
pub fn bus_permutation_trace(rows: usize) -> TraceTable<F, E> {
    assert!(
        rows.is_power_of_two() && rows >= 2,
        "rows must be a power of two ≥ 2"
    );
    let a: Vec<FE> = (0..rows as u64).map(|i| FE::from(i + 1)).collect();
    let b: Vec<FE> = a.iter().rev().cloned().collect();
    let c: Vec<FE> = (0..rows as u64)
        .map(|i| FE::from((i * 7919) % 4099 + 1))
        .collect();
    let d: Vec<FE> = (0..rows).map(|i| c[(i + 1) % rows]).collect();
    TraceTable::from_columns_main(vec![a, b, c, d], 1)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::prover::{IsStarkProver, Prover};
    use crate::traits::AIR;
    use crate::verifier::{IsStarkVerifier, Verifier};
    use crypto::fiat_shamir::default_transcript::DefaultTranscript;

    /// The two properties the device tests rely on: the AIR hands out a
    /// constraint program (the CUDA composition arm's input), and an honest
    /// trace proves and verifies as a single table (the bus balances to zero).
    #[test]
    fn the_bus_permutation_table_round_trips_and_has_a_constraint_program() {
        let options = ProofOptions::default_test_options();
        let air = bus_permutation_air(&options);
        assert!(!air.constraint_program().nodes.is_empty());
        let mut trace = bus_permutation_trace(64);
        let proof = Prover::prove(&air, &mut trace, &(), &mut DefaultTranscript::<E>::new(&[]))
            .expect("an honest trace proves");
        assert!(proof.lde_trace_aux_merkle_root.is_some(), "a LogUp table");
        assert!(Verifier::verify(
            &proof,
            &air,
            &mut DefaultTranscript::<E>::new(&[])
        ));
    }

    /// Non-vacuity of the balance: `d` no longer a permutation of `c` makes the
    /// table's bus contribution non-zero, and the single-table verify refuses.
    #[test]
    fn an_unbalanced_bus_permutation_table_is_rejected() {
        let options = ProofOptions::default_test_options();
        let air = bus_permutation_air(&options);
        let mut trace = bus_permutation_trace(64);
        trace.set_main(5, 3, FE::from(999_999u64));
        let rejected =
            match Prover::prove(&air, &mut trace, &(), &mut DefaultTranscript::<E>::new(&[])) {
                Err(_) => true,
                Ok(proof) => !Verifier::verify(&proof, &air, &mut DefaultTranscript::<E>::new(&[])),
            };
        assert!(rejected, "an unbalanced table must not verify");
    }
}
