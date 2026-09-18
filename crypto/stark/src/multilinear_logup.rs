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
///
/// `candidates` are the columns worth asking about, in ascending order: the
/// coefficient is read by evaluating at each basis vector, so the cost is one
/// pass over the expression per candidate. Handing it the whole table turns
/// that into `interactions × columns`, which for a precompile with a thousand
/// of each is millions of passes for a handful of nonzero terms.
fn probe<E, S>(
    candidates: &[usize],
    slot_of: &mut S,
    eval: impl Fn(&dyn Fn(usize) -> FieldElement<E>) -> FieldElement<E>,
) -> Result<Affine<E>, MlError>
where
    E: IsField + 'static,
    S: FnMut(usize) -> Result<usize, MlError>,
{
    let constant = eval(&|_| FieldElement::<E>::zero());
    let mut terms = Vec::new();
    for &column in candidates {
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
    E: IsField + 'static,
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
            // What this interaction reads, and nothing else: a column it never
            // touches has coefficient zero, and asking costs a pass over the
            // expression.
            let candidates: Vec<usize> = bus
                .columns_read()
                .into_iter()
                .filter(|&column| column < num_main_columns)
                .collect();
            let numerator = probe(&candidates, &mut slot_of, |column| {
                let value = bus.multiplicity.evaluate_with(column);
                if bus.is_sender { value } else { -value }
            })?;
            let denominator = probe(&candidates, &mut slot_of, |column| {
                z - fingerprint_at(bus, &alpha_powers, column)
            })?;
            Ok(Interaction::new(numerator, denominator))
        })
        .collect()
}

/// The same recovery for a VECTOR of linear functions probed together.
///
/// One pass per candidate recovers every component's coefficient at once. That
/// is what [`interaction_shapes`] needs: a fingerprint's bus elements all come
/// out of ONE `combine_from` walk, so probing them one at a time would repeat
/// that walk once per element and turn a pass per candidate into `elements`
/// passes per candidate.
///
/// `slot_of` is asked about a column that is nonzero in ANY component, once —
/// [`probe`]'s laziness widened to the vector.
///
/// ⚠ This duplicates [`probe`]'s dozen lines rather than replacing it.
/// Expressing `probe` as a one-component call here would allocate a `Vec` per
/// eval, and eval is the thing [`interactions`] is written to run as few times
/// as possible. The two are held together by a test instead:
/// `the_vector_probe_is_the_scalar_one_component_by_component`.
fn probe_many<E, S>(
    candidates: &[usize],
    slot_of: &mut S,
    eval: impl Fn(&dyn Fn(usize) -> FieldElement<E>) -> Vec<FieldElement<E>>,
) -> Result<Vec<Affine<E>>, MlError>
where
    E: IsField + 'static,
    S: FnMut(usize) -> Result<usize, MlError>,
{
    let constants = eval(&|_| FieldElement::<E>::zero());
    let mut terms: Vec<Vec<(usize, FieldElement<E>)>> = vec![Vec::new(); constants.len()];
    for &column in candidates {
        let at_basis = eval(&|i| {
            if i == column {
                FieldElement::<E>::one()
            } else {
                FieldElement::<E>::zero()
            }
        });
        // A contract on `eval`, not a runtime condition: the one producer is a
        // fingerprint's bus elements, whose count is a property of the
        // interaction and not of the columns it is asked about. Stated as an
        // assert rather than an error so a future producer that breaks it says
        // so here instead of silently dropping a component.
        assert_eq!(
            at_basis.len(),
            constants.len(),
            "a probed vector must have the same length at every point"
        );
        let coefficients: Vec<FieldElement<E>> = at_basis
            .iter()
            .zip(&constants)
            .map(|(value, constant)| value - constant)
            .collect();
        if coefficients
            .iter()
            .all(|c| *c == FieldElement::<E>::zero())
        {
            continue;
        }
        let slot = slot_of(column)?;
        for (component, coefficient) in coefficients.into_iter().enumerate() {
            if coefficient == FieldElement::zero() {
                continue;
            }
            terms[component].push((slot, coefficient));
        }
    }

    Ok(terms
        .into_iter()
        .zip(constants)
        .map(|(terms, constant)| Affine::new(terms, constant))
        .collect())
}

/// One interaction's affine structure with the LogUp challenges factored OUT.
///
/// [`interactions`] fuses `z` and the powers of `alpha` into the denominator's
/// coefficients, which is right for a verifier that has drawn them and wrong
/// for one that is EMITTING code to be run later: the recursion's in-guest
/// verifier knows the bus at emit time and the challenges only at run time.
/// This is the same bus, cut the other way.
///
/// ```text
///     numerator_i   = the multiplicity, signed          (no challenge at all)
///     denominator_i = z − bus_id − Σ_p alpha^{p+1}·elements[p]
/// ```
///
/// ★ The numerator is not challenge-dependent in any part: `interactions`
/// builds it from `multiplicity.evaluate_with` alone, so it is repeated here
/// verbatim and a caller may intern its coefficients as constants.
pub struct InteractionShape<E: IsField> {
    /// Exactly what [`interactions`] builds, at any `z` and `alpha`.
    pub numerator: Affine<E>,
    /// The fingerprint's `alpha^0` term.
    pub bus_id: FieldElement<E>,
    /// The bus elements in fingerprint order: `elements[p]` rides
    /// `alpha^{p+1}`, matching `fingerprint_at`'s `power` counter.
    pub elements: Vec<Affine<E>>,
}

/// Every interaction's affine structure, challenge-free.
///
/// The same `probe`-the-real-evaluators discipline [`interactions`] uses, and
/// the same `num_main_columns` filter: a column past the main width is PINNED
/// TO ZERO in the recovered affine rather than carried, because it is never a
/// probe candidate.
///
/// ⚠ `slot_of` is asked about a column whose coefficient is structurally
/// nonzero. [`interactions`] asks about a column whose coefficient is nonzero
/// AT ITS ALPHA, which is the same set except on a measure-zero choice of
/// alpha where the powers cancel; the two therefore agree in value always and
/// in term list with overwhelming probability. The test below asserts both.
pub fn interaction_shapes<E, S>(
    buses: &[BusInteraction],
    num_main_columns: usize,
    mut slot_of: S,
) -> Result<Vec<InteractionShape<E>>, MlError>
where
    E: IsField + 'static,
    S: FnMut(usize) -> Result<usize, MlError>,
{
    buses
        .iter()
        .map(|bus| {
            let candidates: Vec<usize> = bus
                .columns_read()
                .into_iter()
                .filter(|&column| column < num_main_columns)
                .collect();
            let numerator = probe(&candidates, &mut slot_of, |column| {
                let value = bus.multiplicity.evaluate_with(column);
                if bus.is_sender { value } else { -value }
            })?;
            let elements = probe_many(&candidates, &mut slot_of, |column| {
                bus.values
                    .iter()
                    .flat_map(|value| value.combine_from(column))
                    .collect()
            })?;
            Ok(InteractionShape {
                numerator,
                bus_id: FieldElement::<E>::from(bus.bus_id),
                elements,
            })
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

    /// Deterministic extension values, one per main column.
    fn column_values(seed: u64, count: usize) -> Vec<ExtE> {
        let mut state = seed | 1;
        let mut next = || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            FE::from(state >> 2)
        };
        (0..count)
            .map(|_| {
                let (a, b, c) = (next(), next(), next());
                ExtE::new([a, b, c])
            })
            .collect()
    }

    /// The affine recovery over a VECTOR agrees, component by component, with
    /// the scalar one it is written beside.
    ///
    /// The two are separate code because `probe` sits on a path whose cost is
    /// the number of eval calls, and expressing it as a one-component
    /// `probe_many` would allocate a `Vec` per call. This test is what holds
    /// them together: a drift in either fires here.
    #[test]
    fn the_vector_probe_is_the_scalar_one_component_by_component() {
        let options = ProofOptions::default_test_options();
        let air = new_cpu_air_with_lookup(&options);
        let mut components = 0;
        for bus in air.bus_interactions() {
            let candidates: Vec<usize> = bus
                .columns_read()
                .into_iter()
                .filter(|&column| column < 5)
                .collect();
            let mut slot_of = columns_as_factors;
            let together: Vec<Affine<Ext>> = probe_many(&candidates, &mut slot_of, |column| {
                bus.values
                    .iter()
                    .flat_map(|value| value.combine_from(column))
                    .collect()
            })
            .unwrap();
            assert!(
                !together.is_empty(),
                "the interaction must have bus elements for this to compare anything"
            );
            for (index, component) in together.iter().enumerate() {
                let mut slot_of = columns_as_factors;
                let alone: Affine<Ext> = probe(&candidates, &mut slot_of, |column| {
                    bus.values
                        .iter()
                        .flat_map(|value| value.combine_from(column))
                        .nth(index)
                        .expect("the element index is inside the interaction")
                })
                .unwrap();
                assert_eq!(component.terms(), alone.terms(), "element {index}'s terms");
                assert_eq!(
                    component.constant_term(),
                    alone.constant_term(),
                    "element {index}'s constant"
                );
                components += 1;
            }
        }
        assert_eq!(components, 6, "two interactions of three bus elements each");
    }

    /// ★ The challenge-free shapes rebuild what `interactions` fuses.
    ///
    /// This is the whole contract: a caller holding the bus at emit time and
    /// the challenges only at run time can put them back together and land on
    /// what the verifier would have computed. A wrong alpha offset, a reversed
    /// element order, a dropped bus id, a lost receiver sign or a dropped
    /// out-of-range filter each break it — and the challenges are varied
    /// because an identity that only holds at one `alpha` is not one.
    #[test]
    fn the_shapes_recombine_into_the_fused_interactions() {
        let options = ProofOptions::default_test_options();
        let cpu = new_cpu_air_with_lookup(&options);
        let add = new_add_air_with_lookup(&options);
        let mul = new_mul_air_with_lookup(&options);
        let cases: Vec<(&str, &[BusInteraction], usize)> = vec![
            ("cpu", cpu.bus_interactions(), 5),
            ("add", add.bus_interactions(), 4),
            ("mul", mul.bus_interactions(), 4),
            // ⚠ The CPU bus reads columns 0..=4; declaring only three main
            // columns is what exercises the `column < num_main_columns`
            // filter, which pins columns 3 and 4 to ZERO on BOTH sides rather
            // than carrying them. Without this case the filter is a line no
            // gate reaches.
            ("cpu narrowed to 3 main columns", cpu.bus_interactions(), 3),
        ];
        for (name, buses, width) in cases {
            let mut slot_of = columns_as_factors;
            let shapes: Vec<InteractionShape<Ext>> =
                interaction_shapes(buses, width, &mut slot_of).unwrap();
            assert_eq!(shapes.len(), buses.len(), "{name}");

            for (index, (z, alpha)) in [
                (ExtE::from(7u64), ExtE::from(11u64)),
                (ExtE::from(0x9E37_79B9u64), ExtE::from(31u64)),
                (ExtE::new([FE::from(3u64), FE::from(5u64), FE::from(9u64)]), ExtE::new([FE::from(2u64), FE::from(0u64), FE::from(4u64)])),
            ]
            .into_iter()
            .enumerate()
            {
                let fused: Vec<Interaction<Ext>> =
                    interactions(buses, width, &z, &alpha, columns_as_factors).unwrap();
                let values = column_values(0xB0_5A11 + index as u64, width);

                for (i, (fused_i, shape)) in fused.iter().zip(&shapes).enumerate() {
                    assert_eq!(
                        fused_i.numerator.evaluate(&values),
                        shape.numerator.evaluate(&values),
                        "{name} interaction {i}: the numerator reads no challenge and must be identical"
                    );

                    // `fingerprint_at`'s own loop: the bus id at alpha^0, then
                    // each element at the next power.
                    let mut power = alpha.clone();
                    let mut fingerprint = shape.bus_id.clone();
                    for element in &shape.elements {
                        fingerprint = fingerprint + &power * element.evaluate(&values);
                        power = &power * &alpha;
                    }
                    assert_eq!(
                        fused_i.denominator.evaluate(&values),
                        &z - &fingerprint,
                        "{name} interaction {i} at challenge set {index}"
                    );

                    // The same statement about the TERM LIST, which the value
                    // comparison cannot make: the slots the fused affine keeps
                    // are exactly the slots some element reads. ⚠ True with
                    // overwhelming probability rather than always — a fused
                    // coefficient is a polynomial in alpha and could vanish at
                    // a particular one — so this is a check on these
                    // challenges, not a theorem.
                    let mut fused_slots: Vec<usize> =
                        fused_i.denominator.terms().iter().map(|(s, _)| *s).collect();
                    fused_slots.sort_unstable();
                    let mut element_slots: Vec<usize> = shape
                        .elements
                        .iter()
                        .flat_map(|e| e.terms().iter().map(|(s, _)| *s))
                        .collect();
                    element_slots.sort_unstable();
                    element_slots.dedup();
                    assert_eq!(fused_slots, element_slots, "{name} interaction {i}'s slots");
                }
            }
        }
    }

    /// The element ORDER and the sign, worked out by hand for this bus.
    ///
    /// `Packing::Direct.columns(&[2, 3, 4])` gives each column its own bus
    /// element, so the CPU sender's fingerprint is
    /// `bus_id + alpha·c2 + alpha²·c3 + alpha³·c4` — the same hand derivation
    /// the fused test above this one makes, stated against the pieces rather
    /// than against the sum, which is where an off-by-one in the power would
    /// otherwise hide behind a matching total.
    #[test]
    fn the_elements_are_the_columns_in_fingerprint_order() {
        let options = ProofOptions::default_test_options();
        let cpu = new_cpu_air_with_lookup(&options);
        let mut slot_of = columns_as_factors;
        let shapes: Vec<InteractionShape<Ext>> =
            interaction_shapes(cpu.bus_interactions(), 5, &mut slot_of).unwrap();
        assert_eq!(shapes.len(), 2);
        for (i, shape) in shapes.iter().enumerate() {
            // A sender's numerator is its multiplicity, unnegated: column `i`
            // with coefficient one and no constant.
            assert_eq!(shape.numerator.terms(), &[(i, ExtE::one())]);
            assert_eq!(shape.numerator.constant_term(), &ExtE::zero());
            assert_eq!(shape.elements.len(), 3);
            for (p, element) in shape.elements.iter().enumerate() {
                assert_eq!(
                    element.terms(),
                    &[(p + 2, ExtE::one())],
                    "element {p} of interaction {i} reads column {}",
                    p + 2
                );
                assert_eq!(element.constant_term(), &ExtE::zero());
            }
        }

        // A RECEIVER's numerator carries the sign, which the senders above
        // cannot show.
        let add = new_add_air_with_lookup(&options);
        let mut slot_of = columns_as_factors;
        let received: Vec<InteractionShape<Ext>> =
            interaction_shapes(add.bus_interactions(), 4, &mut slot_of).unwrap();
        assert_eq!(received[0].numerator.terms(), &[(3, -ExtE::one())]);
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
