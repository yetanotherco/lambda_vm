//! Translates a table's bus interactions into the affine form
//! [`multilinear::logup`] works over.
//!
//! A multiplicity and a fingerprint are both **linear** in the main
//! columns, so each interaction is two affine expressions over the sumcheck's
//! factors and neither needs a column of its own. The coefficients are found by
//! **probing the real evaluators** — the value on an all-zero row is the
//! constant, the value on a basis row is that column's coefficient — rather
//! than re-deriving the packing formulas, which is what keeps this from
//! drifting away from what the prover computes.

use math::field::{element::FieldElement, traits::IsField};
use multilinear::{
    Error as MlError,
    logup::{Affine, Interaction},
};

use crate::lookup::BusInteraction;

/// The interaction's fingerprint from a row of column values: the bus id at
/// `alpha^0`, then each bus element at the next power.
///
/// `alpha_powers[i]` must be `alpha^i`, up to the widest interaction.
fn fingerprint_at<E: IsField>(
    interaction: &BusInteraction,
    alpha_powers: &[FieldElement<E>],
    column: &dyn Fn(usize) -> FieldElement<E>,
) -> FieldElement<E> {
    let mut fingerprint = FieldElement::<E>::from(interaction.bus_id);
    let mut power = 1;
    for value in &interaction.values {
        for element in value.combine_from(column) {
            fingerprint += &alpha_powers[power] * element;
            power += 1;
        }
    }
    fingerprint
}

/// Recovers the affine form of a linear function of the main columns.
///
/// `slot_of` maps a main column to the factor that reads it, and is only asked
/// about columns that turn out to have a nonzero coefficient — so a table's
/// unread columns never need a factor.
fn probe<E, S>(
    num_main_columns: usize,
    slot_of: &mut S,
    eval: impl Fn(&dyn Fn(usize) -> FieldElement<E>) -> FieldElement<E>,
) -> Result<Affine<E>, MlError>
where
    E: IsField,
    S: FnMut(usize) -> Result<usize, MlError>,
{
    let constant = eval(&|_| FieldElement::<E>::zero());
    let mut terms = Vec::new();
    for column in 0..num_main_columns {
        let coefficient = eval(&|i| {
            if i == column {
                FieldElement::<E>::one()
            } else {
                FieldElement::<E>::zero()
            }
        }) - &constant;
        if coefficient == FieldElement::zero() {
            continue;
        }
        terms.push((slot_of(column)?, coefficient));
    }
    Ok(Affine::new(terms, constant))
}

/// Every interaction as a signed numerator over `z − fingerprint`.
///
/// `z` and `alpha` are the LogUp challenges, in the same roles the univariate
/// prover gives them. `slot_of` is asked for the factor reading each main
/// column an interaction actually uses, so a caller can register factors
/// lazily.
pub fn interactions<E, S>(
    buses: &[BusInteraction],
    num_main_columns: usize,
    z: &FieldElement<E>,
    alpha: &FieldElement<E>,
    mut slot_of: S,
) -> Result<Vec<Interaction<E>>, MlError>
where
    E: IsField,
    S: FnMut(usize) -> Result<usize, MlError>,
{
    let width = buses
        .iter()
        .map(BusInteraction::num_bus_elements)
        .max()
        .unwrap_or(0);
    let mut alpha_powers = Vec::with_capacity(width + 1);
    let mut power = FieldElement::<E>::one();
    for _ in 0..=width {
        alpha_powers.push(power.clone());
        power *= alpha;
    }

    buses
        .iter()
        .map(|bus| {
            let numerator = probe(num_main_columns, &mut slot_of, |column| {
                let value = bus.multiplicity.evaluate_with(column);
                if bus.is_sender { value } else { -value }
            })?;
            let denominator = probe(num_main_columns, &mut slot_of, |column| {
                z - fingerprint_at(bus, &alpha_powers, column)
            })?;
            Ok(Interaction::new(numerator, denominator))
        })
        .collect()
}

/// The identity map: factor `i` reads main column `i`.
///
/// What a table with no constraints of its own uses, where the factor list is
/// just its columns.
pub fn columns_as_factors(column: usize) -> Result<usize, MlError> {
    Ok(column)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crypto::fiat_shamir::default_transcript::DefaultTranscript;
    use math::field::{
        extensions_goldilocks::Degree3GoldilocksExtensionField as Ext,
        goldilocks::GoldilocksField as Fp,
    };
    use multilinear::{
        batch::Rule,
        eq::eq_mle,
        gkr::{self, FractionTree},
        logup::{self, input_layer},
        mle::Mle,
    };

    use crate::examples::multi_table_lookup::{
        new_add_air_with_lookup, new_cpu_air_with_lookup, new_mul_air_with_lookup,
    };
    use crate::proof::options::ProofOptions;
    use crate::traits::AIR;

    type FE = FieldElement<Fp>;
    type ExtE = FieldElement<Ext>;

    /// The LogUp challenges. Fixed here; the protocol draws them after the
    /// trace is committed.
    fn challenges() -> (ExtE, ExtE) {
        (ExtE::from(0x9E37_79B9u64), ExtE::from(31u64))
    }

    fn base(values: &[u64]) -> Vec<FE> {
        values.iter().map(|v| FE::from(*v)).collect()
    }

    /// One factor per main column, lifted into the extension.
    fn factors(columns: &[Vec<FE>]) -> Vec<Mle<Ext>> {
        columns
            .iter()
            .map(|c| {
                Mle::new(
                    c.iter()
                        .map(|v| v.to_extension::<Ext>())
                        .collect::<Vec<_>>(),
                )
                .unwrap()
            })
            .collect()
    }

    /// The CPU trace from the multi-table completeness test: eight rows
    /// dispatching four additions and four multiplications.
    fn cpu_columns() -> Vec<Vec<FE>> {
        vec![
            base(&[1, 0, 1, 0, 1, 1, 0, 0]),            // add flag
            base(&[0, 1, 0, 1, 0, 0, 1, 1]),            // mul flag
            base(&[1, 2, 3, 4, 5, 6, 7, 8]),            // a
            base(&[10, 20, 30, 40, 50, 60, 70, 80]),    // b
            base(&[11, 40, 33, 160, 55, 66, 490, 640]), // c
        ]
    }

    /// The additions the CPU dispatched, each received once.
    fn add_columns() -> Vec<Vec<FE>> {
        vec![
            base(&[1, 3, 5, 6]),
            base(&[10, 30, 50, 60]),
            base(&[11, 33, 55, 66]),
            base(&[1, 1, 1, 1]),
        ]
    }

    /// The multiplications, likewise.
    fn mul_columns() -> Vec<Vec<FE>> {
        vec![
            base(&[2, 4, 7, 8]),
            base(&[20, 40, 70, 80]),
            base(&[40, 160, 490, 640]),
            base(&[1, 1, 1, 1]),
        ]
    }

    /// `p/q` of a table's whole bus contribution.
    fn contribution(interactions: &[Interaction<Ext>], columns: &[Vec<FE>]) -> ExtE {
        let tree =
            FractionTree::build(input_layer(interactions, &factors(columns)).unwrap()).unwrap();
        let (p, q) = tree.output();
        p * q.inv().unwrap()
    }

    fn cpu_interactions() -> Vec<Interaction<Ext>> {
        let (z, alpha) = challenges();
        let options = ProofOptions::default_test_options();
        let air = new_cpu_air_with_lookup(&options);
        interactions(air.bus_interactions(), 5, &z, &alpha, columns_as_factors).unwrap()
    }

    fn add_interactions() -> Vec<Interaction<Ext>> {
        let (z, alpha) = challenges();
        let options = ProofOptions::default_test_options();
        let air = new_add_air_with_lookup(&options);
        interactions(air.bus_interactions(), 4, &z, &alpha, columns_as_factors).unwrap()
    }

    fn mul_interactions() -> Vec<Interaction<Ext>> {
        let (z, alpha) = challenges();
        let options = ProofOptions::default_test_options();
        let air = new_mul_air_with_lookup(&options);
        interactions(air.bus_interactions(), 4, &z, &alpha, columns_as_factors).unwrap()
    }

    /// The affine form must agree with the formula the univariate prover uses,
    /// worked out by hand for this packing: `Direct.columns(&[2, 3, 4])` puts
    /// each column in its own bus element, so the fingerprint is
    /// `bus_id + alpha·a + alpha²·b + alpha³·c`.
    #[test]
    fn the_affine_form_matches_the_fingerprint_by_hand() {
        let (z, alpha) = challenges();
        let columns = cpu_columns();
        let cpu = cpu_interactions();
        let factors = factors(&columns);
        assert_eq!(cpu.len(), 2);

        for (bus_id, interaction) in cpu.iter().enumerate() {
            for row in 0..columns[0].len() {
                let values: Vec<ExtE> = factors.iter().map(|f| f.evals()[row]).collect();
                let a = values[2];
                let b = values[3];
                let c = values[4];
                let fingerprint = ExtE::from(bus_id as u64)
                    + alpha * a
                    + alpha * alpha * b
                    + alpha * alpha * alpha * c;

                assert_eq!(
                    interaction.denominator.evaluate(&values),
                    z - fingerprint,
                    "bus {bus_id}, row {row}"
                );
                // Both CPU interactions are senders, reading their flag column.
                assert_eq!(
                    interaction.numerator.evaluate(&values),
                    values[bus_id],
                    "bus {bus_id}, row {row}"
                );
            }
        }
    }

    /// A receiver's sign is in its numerator, so the same fingerprint appears
    /// on both sides of the bus with opposite multiplicity.
    #[test]
    fn a_receiver_carries_the_negative_multiplicity() {
        let columns = add_columns();
        let factors = factors(&columns);
        let add = add_interactions();
        assert_eq!(add.len(), 1);

        for row in 0..columns[0].len() {
            let values: Vec<ExtE> = factors.iter().map(|f| f.evals()[row]).collect();
            assert_eq!(add[0].numerator.evaluate(&values), -values[3]);
        }
    }

    /// The milestone: the buses of a real three-table example balance when
    /// computed entirely through the multilinear path — one fraction tree per
    /// table, over tables of different heights.
    #[test]
    fn three_real_tables_balance_across_their_buses() {
        let total = contribution(&cpu_interactions(), &cpu_columns())
            + contribution(&add_interactions(), &add_columns())
            + contribution(&mul_interactions(), &mul_columns());
        assert_eq!(total, ExtE::zero());
    }

    #[test]
    fn a_receive_that_never_happened_unbalances_the_bus() {
        let mut columns = add_columns();
        columns[3][2] = FE::zero(); // one addition goes unreceived

        let total = contribution(&cpu_interactions(), &cpu_columns())
            + contribution(&add_interactions(), &columns)
            + contribution(&mul_interactions(), &mul_columns());
        assert_ne!(total, ExtE::zero());
    }

    #[test]
    fn a_value_that_was_never_sent_unbalances_the_bus() {
        let mut columns = add_columns();
        columns[2][1] += FE::one(); // the received sum is not the one dispatched

        let total = contribution(&cpu_interactions(), &cpu_columns())
            + contribution(&add_interactions(), &columns)
            + contribution(&mul_interactions(), &mul_columns());
        assert_ne!(total, ExtE::zero());
    }

    /// And the claim GKR leaves about a real table's input layer is what two
    /// rules over that table's columns sum to — the whole point of keeping the
    /// numerator and denominator out of the commitment.
    #[test]
    fn a_real_tables_input_claim_reduces_to_two_rules() {
        let columns = cpu_columns();
        let cpu = cpu_interactions();
        let factors = factors(&columns);
        let num_row_vars = factors[0].num_vars();

        let tree = FractionTree::build(input_layer(&cpu, &factors).unwrap()).unwrap();
        let mut transcript = DefaultTranscript::<Ext>::new(b"logup-bridge");
        let out = gkr::prove(&tree, &mut transcript).unwrap();

        let weight = factors.len();
        let statements =
            logup::claim_statements(&cpu, &out.claim.point, num_row_vars, weight).unwrap();
        let mut with_weight = factors;
        with_weight.push(eq_mle(&statements.row_point).unwrap());

        let sum = |rule: &Rule<'_, Ext>| {
            (0..(1usize << num_row_vars)).fold(ExtE::zero(), |acc, i| {
                let values: Vec<ExtE> = with_weight.iter().map(|f| f.evals()[i]).collect();
                acc + rule.apply(&values)
            })
        };
        assert_eq!(sum(&statements.numerator), out.claim.p);
        assert_eq!(sum(&statements.denominator), out.claim.q);
    }

    /// Columns no interaction reads never need a factor: the probe only asks
    /// about the ones with a nonzero coefficient.
    #[test]
    fn unread_columns_are_never_asked_for() {
        let (z, alpha) = challenges();
        let options = ProofOptions::default_test_options();
        let air = new_mul_air_with_lookup(&options);
        // The MUL table reads columns 0, 1, 2 and 3 — so widen it and check the
        // extra column is not demanded.
        let result = interactions(air.bus_interactions(), 6, &z, &alpha, |column| {
            if column >= 4 {
                panic!("column {column} is not read by any interaction");
            }
            Ok(column)
        });
        assert!(result.is_ok());
    }
}
