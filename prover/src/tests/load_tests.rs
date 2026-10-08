//! Tests for the LOAD table.

use crate::tables::load::*;

#[test]
fn test_load_trace_generation() {
    // Load 4 bytes, sign-extend
    let ops = vec![
        LoadOperation::new(
            0x1000,
            100,
            4,
            true,
            [0x12, 0x34, 0x56, 0x78, 0xFF, 0xFF, 0xFF, 0xFF],
        ),
        LoadOperation::new(
            0x2000,
            200,
            1,
            false,
            [0x42, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00],
        ),
    ];

    let trace = generate_load_trace(&ops);
    assert_eq!(trace.num_cols(), cols::NUM_COLUMNS);
    assert!(trace.num_rows() >= 2);
}

#[test]
fn test_read_flags() {
    // "Exactly N" semantics per spec
    let op1 = LoadOperation::new(0, 0, 1, false, [0; 8]);
    assert_eq!(op1.read_flags(), (false, false, false)); // no flags for 1 byte

    let op2 = LoadOperation::new(0, 0, 2, false, [0; 8]);
    assert_eq!(op2.read_flags(), (true, false, false)); // read2 only

    let op4 = LoadOperation::new(0, 0, 4, false, [0; 8]);
    assert_eq!(op4.read_flags(), (false, true, false)); // read4 only

    let op8 = LoadOperation::new(0, 0, 8, false, [0; 8]);
    assert_eq!(op8.read_flags(), (false, false, true)); // read8 only
}

#[test]
fn test_sign_bit_extraction() {
    // Byte with MSB set
    let op1 = LoadOperation::new(0, 0, 1, true, [0x80, 0, 0, 0, 0, 0, 0, 0]);
    assert!(op1.compute_sign_bit());

    // Byte without MSB set
    let op2 = LoadOperation::new(0, 0, 1, true, [0x7F, 0, 0, 0, 0, 0, 0, 0]);
    assert!(!op2.compute_sign_bit());

    // Halfword with MSB set
    let op3 = LoadOperation::new(0, 0, 2, true, [0x00, 0x80, 0, 0, 0, 0, 0, 0]);
    assert!(op3.compute_sign_bit());

    // Word with MSB set
    let op4 = LoadOperation::new(0, 0, 4, true, [0, 0, 0, 0x80, 0, 0, 0, 0]);
    assert!(op4.compute_sign_bit());
}

// =========================================================================
// Soundness regression: μ is bit-constrained (MuIsBit, idx 13).
//
// Before MuIsBit, idx 5 pinned μ = 1 only when a read2/4/8 flag was set, so a
// byte-load row's μ was free. A μ = −1 / μ = 1 pair of LBU rows with the same
// `res[0]` cancels on MEMORY/MEMW but nets `MSB8[res[0]] → 0` minus `→ 1`,
// which cancels a real LB row's forged `sign_bit = 1`: `lb 0x05` answered
// `0xFFFF_FFFF_FFFF_FF05` and verified end to end
// (`multiplicity_forgery_poc::load_negative_mu_sign_extension_forgery_is_rejected`).
// =========================================================================

mod mu_bit_regression {
    use crate::tables::load::{LoadConstraints, LoadOperation, cols, generate_load_trace};
    use crate::tables::types::{FE, GoldilocksExtension, GoldilocksField};
    use crate::test_utils::{busless_air, validate_busless};

    type Trace = stark::trace::TraceTable<GoldilocksField, GoldilocksExtension>;

    /// Row 0: the real `lb` of 0x05 with the forged sign extension. Rows 1/2:
    /// the cancelling `μ = −1 / μ = 1` byte-load pair.
    fn forged_lb() -> (Trace, Trace) {
        let op = LoadOperation::new(0x1000, 100, 1, true, [0x05, 0, 0, 0, 0, 0, 0, 0]);
        let honest = generate_load_trace(std::slice::from_ref(&op));
        let mut forged = honest.clone();
        let t = &mut forged.main_table;
        assert_eq!(*t.get(0, cols::SIGN_BIT), FE::zero());
        for &c in &cols::RES[1..] {
            t.set(0, c, FE::from(0xFFu64));
        }
        t.set(0, cols::SIGN_BIT, FE::one());
        for (row, mu, sign_bit) in [(1, -FE::one(), FE::one()), (2, FE::one(), FE::zero())] {
            t.set(row, cols::RES[0], FE::from(0x05u64));
            t.set(row, cols::MU, mu);
            t.set(row, cols::SIGN_BIT, sign_bit);
        }
        (honest, forged)
    }

    #[test]
    fn negative_mu_byte_load_pair_is_rejected() {
        let (honest, forged) = forged_lb();
        let air = busless_air(cols::NUM_COLUMNS, LoadConstraints);
        assert!(validate_busless(&air, &honest), "honest trace must pass");
        assert!(
            !validate_busless(&air, &forged),
            "μ = −1 must be rejected by IS_BIT[μ]"
        );
    }

    /// Every other in-chip constraint holds on the forged rows: only MuIsBit
    /// stands between the trace and a forged sign extension.
    #[test]
    fn forged_rows_differ_from_valid_only_in_mu() {
        let (_, mut forged) = forged_lb();
        forged.main_table.set(1, cols::MU, FE::zero());
        forged.main_table.set(2, cols::MU, FE::zero());
        let air = busless_air(cols::NUM_COLUMNS, LoadConstraints);
        assert!(validate_busless(&air, &forged));
    }

    #[test]
    fn mu_two_byte_load_row_is_rejected() {
        let op = LoadOperation::new(0x1000, 100, 1, false, [0x42, 0, 0, 0, 0, 0, 0, 0]);
        let mut trace = generate_load_trace(std::slice::from_ref(&op));
        trace.main_table.set(0, cols::MU, FE::from(2u64));
        let air = busless_air(cols::NUM_COLUMNS, LoadConstraints);
        assert!(
            !validate_busless(&air, &trace),
            "μ = 2 must be rejected by IS_BIT[μ]"
        );
    }
}
