//! Tests for the SHIFT table.

use stark::proof::options::ProofOptions;
use stark::traits::AIR;

use crate::tables::shift::{NUM_SHIFT_CONSTRAINTS, bus_interactions, cols};
use crate::test_utils::{create_shift_air, in_chip_constraint_count, is_halfword_sender_columns};

// Soundness regression: every input halfword must be range-checked. That gap was
// closed by the in-chip soundness fix (VM-3). The `(1 - signed) * is_negative = 0`
// constraint (VM-4) is a spec-level gap (`shift.toml` omits it) and is deferred to a
// separate spec fix, so it is intentionally not enforced here.

/// Presence: every input halfword is range-checked via IS_HALFWORD senders, so a
/// field-wrapping decomposition that keeps the packed operand constant cannot
/// change the shifted output undetected.
#[test]
fn test_shift_range_checks_input_halves() {
    let cols_checked = is_halfword_sender_columns(&bus_interactions());
    for c in cols::IN {
        assert!(
            cols_checked.contains(&c),
            "SHIFT must IS_HALF range-check input half column {c}"
        );
    }
}

/// Wiring: `create_shift_air` registers all in-chip constraints on top of its bus
/// constraints. Catches a revert to `transition_constraints = vec![]` or a dropped
/// constraint (the count would differ).
#[test]
fn test_shift_air_wires_in_chip_constraints() {
    let air = create_shift_air(&ProofOptions::default_test_options());
    let in_chip = in_chip_constraint_count(
        air.num_transition_constraints(),
        cols::NUM_COLUMNS,
        bus_interactions(),
    );
    assert_eq!(in_chip, NUM_SHIFT_CONSTRAINTS);
    // 19 + ZbsIsBit + MuIsBit.
    assert_eq!(NUM_SHIFT_CONSTRAINTS, 21);
}

// =========================================================================
// Soundness regression: `zbs` and `μ` are bit-constrained.
//
// Before ZbsIsBit (idx 19) nothing pinned `zbs` on a μ = 0 padding row, so
// `zbs = 2` made the `1 - zbs` HWSL multiplicity −1 and the row provided the
// HWSL tuples a forged real row needed (`SLL 1, 1 = 0`, verified on ffc4ac19).
// Before MuIsBit (idx 20) a left shift's output scaled with a free μ.
// =========================================================================

mod zbs_mu_bit_regression {
    use std::collections::HashMap;

    use math::field::element::FieldElement;
    use stark::lookup::{LinearTerm, Multiplicity};
    use stark::trace::TraceTable;

    use crate::tables::shift::{
        ShiftConstraints, ShiftOperation, bus_interactions, cols, generate_shift_trace,
    };
    use crate::tables::types::{BusId, GoldilocksExtension, GoldilocksField};
    use crate::test_utils::{busless_air, validate_busless};

    type FE = FieldElement<GoldilocksField>;
    type Trace = TraceTable<GoldilocksField, GoldilocksExtension>;

    fn mult(m: &Multiplicity, g: &dyn Fn(usize) -> FE) -> FE {
        match m {
            Multiplicity::One => FE::one(),
            Multiplicity::Column(c) => g(*c),
            Multiplicity::Sum(a, b) => g(*a) + g(*b),
            Multiplicity::Negated(c) => FE::one() - g(*c),
            Multiplicity::Diff(a, b) => g(*a) - g(*b),
            Multiplicity::Sum3(a, b, c) => g(*a) + g(*b) + g(*c),
            Multiplicity::Linear(ts) => ts.iter().fold(FE::zero(), |acc, t| match *t {
                LinearTerm::Column {
                    coefficient,
                    column,
                } => acc + g(column) * FE::from(coefficient),
                LinearTerm::ColumnUnsigned {
                    coefficient,
                    column,
                } => acc + g(column) * FE::from(coefficient),
                LinearTerm::Constant(v) => acc + FE::from(v),
            }),
        }
    }

    /// Net multiset contribution of the SHIFT table per (bus, tuple): sender +m,
    /// receiver −m, zero entries dropped. This is exactly what LogUp checks.
    fn bus_net(trace: &Trace) -> HashMap<(u64, Vec<u64>), FE> {
        let t = &trace.main_table;
        let mut net: HashMap<(u64, Vec<u64>), FE> = HashMap::new();
        for bi in bus_interactions() {
            for r in 0..trace.num_rows() {
                let g = |c: usize| *t.get(r, c);
                let m = mult(&bi.multiplicity, &g);
                if m == FE::zero() {
                    continue;
                }
                let tuple: Vec<u64> = bi
                    .values
                    .iter()
                    .flat_map(|v| v.combine_from(g))
                    .map(|x: FE| x.canonical_u64())
                    .collect();
                let e = net.entry((bi.bus_id, tuple)).or_insert(FE::zero());
                *e = if bi.is_sender { *e + m } else { *e - m };
            }
        }
        net.retain(|_, v| *v != FE::zero());
        net
    }

    fn only(net: &HashMap<(u64, Vec<u64>), FE>, bus: BusId) -> HashMap<Vec<u64>, FE> {
        let id: u64 = bus.into();
        net.iter()
            .filter(|((b, _), _)| *b == id)
            .map(|((_, t), m)| (t.clone(), *m))
            .collect()
    }

    /// Forges `SLL 1, 1 = 0` (truth: 2) on the SHIFT chip.
    ///
    /// Row 0 (μ=1, real): honest except `X[0] = 0` → `out = 0`; its HWSL tuple
    /// `[1, 1, 0, 0]` is not in the HWSL table.
    /// Row 1 (padding, μ=0): `zbs = 2`, `in[0] = 1`, `bit_shift = 1`. With μ=0
    /// every other lookup and `left/right` vanish, so the zbs-override constraints
    /// only force `X = Y = 0`; the five HWSL senders then fire with multiplicity
    /// `1 - zbs = -1` and exactly cancel row 0's five HWSL tuples.
    fn forged_sll() -> (ShiftOperation, Trace, Trace) {
        let op = ShiftOperation::new(1, 1, false, false, false);
        let honest = generate_shift_trace(std::slice::from_ref(&op));
        let mut forged = honest.clone();
        {
            let t = &mut forged.main_table;
            assert_eq!(*t.get(0, cols::X_0), FE::from(2u64));
            t.set(0, cols::X_0, FE::zero());
            t.set(0, cols::OUT_0, FE::zero());

            t.set(1, cols::ZBS, FE::from(2u64));
            t.set(1, cols::IN_0, FE::one());
            t.set(1, cols::BIT_SHIFT, FE::one());
        }
        (op, honest, forged)
    }

    /// The forgery is complete on every bus: if the trace passed the in-chip
    /// constraints, `1 << 1 = 0` would be accepted. This is what makes
    /// `zbs_two_padding_row_is_rejected` load-bearing.
    #[test]
    fn forged_sll_would_balance_every_bus() {
        let (op, honest, forged) = forged_sll();
        let (h, f) = (bus_net(&honest), bus_net(&forged));

        // HWSL: the forged trace needs *no* HWSL table entries at all.
        assert!(!only(&h, BusId::Hwsl).is_empty());
        assert!(only(&f, BusId::Hwsl).is_empty(), "HWSL fully self-cancels");

        // Every other lookup bus is identical to the honest run (all real entries).
        for bus in [
            BusId::ByteAlu,
            BusId::Zero,
            BusId::AreBytes,
            BusId::IsHalfword,
            BusId::Msb16,
        ] {
            assert_eq!(only(&h, bus), only(&f, bus), "{bus:?} demand unchanged");
        }

        // ALU: the chip would accept the CPU's claim `1 << 1 = 0`.
        let alu = only(&f, BusId::Alu);
        assert_eq!(alu.len(), 1);
        let (tuple, m) = alu.iter().next().unwrap();
        assert_eq!(*m, -FE::one(), "received once");
        let out = tuple[tuple.len() - 2] | (tuple[tuple.len() - 1] << 32);
        assert_eq!(op.compute_out(), 2);
        assert_eq!(out, 0, "forged result");
    }

    #[test]
    fn zbs_two_padding_row_is_rejected() {
        let (_, honest, forged) = forged_sll();
        let air = busless_air(cols::NUM_COLUMNS, ShiftConstraints);
        assert!(validate_busless(&air, &honest), "honest trace must pass");
        assert!(
            !validate_busless(&air, &forged),
            "zbs = 2 must be rejected by IS_BIT[zbs]"
        );
    }

    /// A left shift with μ = 2 claims twice the result (`left = μ - direction`
    /// scales `out`); it would answer two identical CPU lookups with `2·(x << s)`.
    #[test]
    fn mu_two_left_shift_row_is_rejected() {
        let op = ShiftOperation::new(1, 1, false, false, false);
        let mut trace = generate_shift_trace(std::slice::from_ref(&op));
        {
            let t = &mut trace.main_table;
            assert_eq!(*t.get(0, cols::OUT_0), FE::from(2u64));
            t.set(0, cols::MU, FE::from(2u64));
            t.set(0, cols::OUT_0, FE::from(4u64));
        }
        let air = busless_air(cols::NUM_COLUMNS, ShiftConstraints);
        assert!(
            !validate_busless(&air, &trace),
            "μ = 2 must be rejected by IS_BIT[μ]"
        );
    }
}
