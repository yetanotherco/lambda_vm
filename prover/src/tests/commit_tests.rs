//! Tests for the COMMIT (ECALL) table.
//!
//! Covers trace generation, constraint formula verification, and edge cases.

use crate::constraints::templates::INV_SHIFT_32;
use crate::tables::commit::{CommitOperation, cols, generate_commit_trace};
use crate::tables::types::FE;

// =========================================================================
// Helper: build a commit row
// =========================================================================

fn op(
    timestamp: u64,
    index: u64,
    address: u64,
    count: u64,
    first: bool,
    end: bool,
    value: u8,
) -> CommitOperation {
    CommitOperation {
        timestamp,
        index,
        address,
        count,
        first,
        end,
        value,
    }
}

// =========================================================================
// Trace generation tests
// =========================================================================

#[test]
fn test_commit_single_byte() {
    // count=1: first row (first=1, count=1, value=0x41) + end row (end=1, count=0)
    let ops = vec![
        op(100, 0, 0x1000, 1, true, false, 0x41),
        op(100, 1, 0x1001, 0, false, true, 0),
    ];
    let trace = generate_commit_trace(&ops);

    // Row 0: first=1, end=0, count=1, value=0x41, mu=1
    let r0 = trace.main_table.get_row(0);
    assert_eq!(r0[cols::FIRST], FE::one());
    assert_eq!(r0[cols::END], FE::zero());
    assert_eq!(r0[cols::COUNT_0], FE::one());
    assert_eq!(r0[cols::COUNT_1], FE::zero());
    assert_eq!(r0[cols::VALUE], FE::from(0x41u64));
    assert_eq!(r0[cols::MU], FE::one());
    assert_eq!(r0[cols::TIMESTAMP], FE::from(100u64));
    assert_eq!(r0[cols::INDEX], FE::zero());

    // Row 0: address = 0x1000
    assert_eq!(r0[cols::ADDRESS_0], FE::from(0x1000u64));
    assert_eq!(r0[cols::ADDRESS_1], FE::zero());

    // Row 0: address_incr = 0x1001
    assert_eq!(r0[cols::ADDRESS_INCR_0], FE::from(0x1001u64));
    assert_eq!(r0[cols::ADDRESS_INCR_1], FE::zero());
    assert_eq!(r0[cols::ADDRESS_INCR_2], FE::zero());
    assert_eq!(r0[cols::ADDRESS_INCR_3], FE::zero());

    // Row 0: count_decr = 0 (count=1 → count-1=0)
    assert_eq!(r0[cols::COUNT_DECR_0], FE::zero());
    assert_eq!(r0[cols::COUNT_DECR_1], FE::zero());
    assert_eq!(r0[cols::COUNT_DECR_2], FE::zero());
    assert_eq!(r0[cols::COUNT_DECR_3], FE::zero());

    // Row 1: first=0, end=1, count=0, value=0, mu=1
    let r1 = trace.main_table.get_row(1);
    assert_eq!(r1[cols::FIRST], FE::zero());
    assert_eq!(r1[cols::END], FE::one());
    assert_eq!(r1[cols::COUNT_0], FE::zero());
    assert_eq!(r1[cols::VALUE], FE::zero());
    assert_eq!(r1[cols::MU], FE::one());
    assert_eq!(r1[cols::INDEX], FE::one());

    // Row 1: count_decr = all 0xFFFF (count=0 → underflow)
    assert_eq!(r1[cols::COUNT_DECR_0], FE::from(0xFFFFu64));
    assert_eq!(r1[cols::COUNT_DECR_1], FE::from(0xFFFFu64));
    assert_eq!(r1[cols::COUNT_DECR_2], FE::from(0xFFFFu64));
    assert_eq!(r1[cols::COUNT_DECR_3], FE::from(0xFFFFu64));
}

#[test]
fn test_commit_multi_byte() {
    // count=3: 3 data rows + 1 end row = 4 rows
    let ops = vec![
        op(200, 10, 0x2000, 3, true, false, b'H'),
        op(200, 11, 0x2001, 2, false, false, b'i'),
        op(200, 12, 0x2002, 1, false, false, b'!'),
        op(200, 13, 0x2003, 0, false, true, 0),
    ];
    let trace = generate_commit_trace(&ops);

    // Row 0: first=1
    let r0 = trace.main_table.get_row(0);
    assert_eq!(r0[cols::FIRST], FE::one());
    assert_eq!(r0[cols::END], FE::zero());
    assert_eq!(r0[cols::COUNT_0], FE::from(3u64));
    assert_eq!(r0[cols::VALUE], FE::from(b'H' as u64));
    assert_eq!(r0[cols::INDEX], FE::from(10u64));

    // Row 1: middle row, count decrement 3→2
    let r1 = trace.main_table.get_row(1);
    assert_eq!(r1[cols::FIRST], FE::zero());
    assert_eq!(r1[cols::END], FE::zero());
    assert_eq!(r1[cols::COUNT_0], FE::from(2u64));
    assert_eq!(r1[cols::VALUE], FE::from(b'i' as u64));
    assert_eq!(r1[cols::INDEX], FE::from(11u64));

    // Row 2: middle row, count decrement 2→1
    let r2 = trace.main_table.get_row(2);
    assert_eq!(r2[cols::COUNT_0], FE::from(1u64));
    assert_eq!(r2[cols::VALUE], FE::from(b'!' as u64));
    assert_eq!(r2[cols::INDEX], FE::from(12u64));

    // Row 3: end row
    let r3 = trace.main_table.get_row(3);
    assert_eq!(r3[cols::FIRST], FE::zero());
    assert_eq!(r3[cols::END], FE::one());
    assert_eq!(r3[cols::COUNT_0], FE::zero());
    assert_eq!(r3[cols::INDEX], FE::from(13u64));

    // All rows share timestamp and mu=1
    for row in 0..4 {
        let r = trace.main_table.get_row(row);
        assert_eq!(r[cols::TIMESTAMP], FE::from(200u64));
        assert_eq!(r[cols::MU], FE::one());
    }

    // Address chain: 0x2000, 0x2001, 0x2002, 0x2003
    for (row, addr) in (0x2000u64..=0x2003).enumerate() {
        let r = trace.main_table.get_row(row);
        assert_eq!(r[cols::ADDRESS_0], FE::from(addr));
    }
}

#[test]
fn test_commit_zero_count() {
    // count=0: single row with first=1 AND end=1
    let ops = vec![op(50, 7, 0x3000, 0, true, true, 0)];
    let trace = generate_commit_trace(&ops);
    let r0 = trace.main_table.get_row(0);

    assert_eq!(r0[cols::FIRST], FE::one());
    assert_eq!(r0[cols::END], FE::one());
    assert_eq!(r0[cols::COUNT_0], FE::zero());
    assert_eq!(r0[cols::MU], FE::one());
    assert_eq!(r0[cols::INDEX], FE::from(7u64));

    // count_decr = all 0xFFFF when count=0
    assert_eq!(r0[cols::COUNT_DECR_0], FE::from(0xFFFFu64));
    assert_eq!(r0[cols::COUNT_DECR_1], FE::from(0xFFFFu64));
    assert_eq!(r0[cols::COUNT_DECR_2], FE::from(0xFFFFu64));
    assert_eq!(r0[cols::COUNT_DECR_3], FE::from(0xFFFFu64));
}

#[test]
fn test_commit_trace_padding() {
    // 1 real row → padded to 4 (minimum power of 2)
    let ops = vec![op(10, 0, 0x100, 0, true, true, 0)];
    let trace = generate_commit_trace(&ops);
    assert_eq!(trace.num_rows(), 4);

    // Padding rows (1..4): mu=0, count=1, address_incr_0=1
    for row in 1..4 {
        let r = trace.main_table.get_row(row);
        assert_eq!(r[cols::MU], FE::zero());
        assert_eq!(r[cols::COUNT_0], FE::one());
        assert_eq!(r[cols::ADDRESS_INCR_0], FE::one());
        assert_eq!(r[cols::FIRST], FE::zero());
        assert_eq!(r[cols::END], FE::zero());
        assert_eq!(r[cols::VALUE], FE::zero());
        assert_eq!(r[cols::ADDRESS_0], FE::zero());
        assert_eq!(r[cols::TIMESTAMP], FE::zero());
        assert_eq!(r[cols::INDEX], FE::zero());
    }
}

#[test]
fn test_commit_trace_dimensions() {
    // 5 rows → next power of 2 = 8
    let ops: Vec<_> = (0..5)
        .map(|i| op(300, i, 0x4000 + i, 5 - i, i == 0, i == 4, (0x60 + i) as u8))
        .collect();
    let trace = generate_commit_trace(&ops);

    assert_eq!(trace.num_rows(), 8);
    assert_eq!(cols::NUM_COLUMNS, 18);
}

// =========================================================================
// Constraint formula tests (field arithmetic)
// =========================================================================

#[test]
fn test_is_bit_constraints() {
    // x * (1 - x) = 0 for x in {0, 1}
    for x_val in [FE::zero(), FE::one()] {
        let result = x_val * (FE::one() - x_val);
        assert_eq!(result, FE::zero());
    }
    // x=2 should fail
    let x = FE::from(2u64);
    assert_ne!(x * (FE::one() - x), FE::zero());
}

#[test]
fn test_first_or_end_implies_mu() {
    // (first + end) * (1 - mu) = 0
    // Valid combos: (0,0,0), (0,0,1), (1,0,1), (0,1,1), (1,1,1)
    let valid = [
        (0u64, 0u64, 0u64),
        (0, 0, 1),
        (1, 0, 1),
        (0, 1, 1),
        (1, 1, 1),
    ];
    for (f, e, m) in valid {
        let first = FE::from(f);
        let end = FE::from(e);
        let mu = FE::from(m);
        let result = (first + end) * (FE::one() - mu);
        assert_eq!(
            result,
            FE::zero(),
            "Should pass for first={f}, end={e}, mu={m}"
        );
    }

    // Invalid: first=1, mu=0
    let result = (FE::one() + FE::zero()) * (FE::one() - FE::zero());
    assert_ne!(result, FE::zero());

    // Invalid: end=1, mu=0
    let result = (FE::zero() + FE::one()) * (FE::one() - FE::zero());
    assert_ne!(result, FE::zero());
}

#[test]
fn test_add_constraint_address() {
    // address + 1 = address_incr
    // carry_0 = (addr_lo + 1 - incr_lo) * 2^(-32)
    let inv_2_32 = FE::from(INV_SHIFT_32);

    // Case 1: no carry. address=0x1000, address+1=0x1001
    let addr_lo = FE::from(0x1000u64);
    let incr_lo = FE::from(0x1001u64);
    let carry_0 = (addr_lo + FE::one() - incr_lo) * inv_2_32;
    assert_eq!(carry_0, FE::zero());
    assert_eq!(carry_0 * (FE::one() - carry_0), FE::zero());

    // carry_1 = (addr_hi + carry_0 - incr_hi) * 2^(-32)
    let carry_1 = (FE::zero() + carry_0 - FE::zero()) * inv_2_32;
    assert_eq!(carry_1, FE::zero());

    // Case 2: carry at 32-bit boundary. address=0x0000_0000_FFFF_FFFF
    // address+1 = 0x0000_0001_0000_0000
    // DWordHL halfwords: [0x0000, 0x0000, 0x0001, 0x0000]
    // incr_lo = h[0] + 2^16*h[1] = 0
    // incr_hi = h[2] + 2^16*h[3] = 1
    let addr_lo_2 = FE::from(0xFFFF_FFFFu64);
    let incr_lo_2 = FE::zero();
    let incr_hi_2 = FE::one();
    let carry_0_2 = (addr_lo_2 + FE::one() - incr_lo_2) * inv_2_32;
    assert_eq!(carry_0_2, FE::one());
    let carry_1_2 = (FE::zero() + carry_0_2 - incr_hi_2) * inv_2_32;
    assert_eq!(carry_1_2, FE::zero());
}

#[test]
fn test_sub_constraint_count() {
    // SUB via reversed ADD: count_decr + 1 = count
    // carry_0 = (count_decr_lo + 1 - count_lo) * 2^(-32)
    let inv_2_32 = FE::from(INV_SHIFT_32);

    // Case 1: count=3, count_decr=2
    let cd_lo = FE::from(2u64);
    let count_lo = FE::from(3u64);
    let carry_0 = (cd_lo + FE::one() - count_lo) * inv_2_32;
    assert_eq!(carry_0, FE::zero());

    // Case 2: count=0, count_decr=0xFFFF_FFFF_FFFF_FFFF
    // count_decr_lo = 0xFFFF + 0xFFFF*2^16 = 0xFFFF_FFFF
    let cd_lo_0 = FE::from(0xFFFF_FFFFu64);
    let cd_hi_0 = FE::from(0xFFFF_FFFFu64);
    // carry_0 = (0xFFFF_FFFF + 1 - 0) * 2^(-32) = 1
    let carry_0_0 = (cd_lo_0 + FE::one() - FE::zero()) * inv_2_32;
    assert_eq!(carry_0_0, FE::one());
    // carry_1 = (0xFFFF_FFFF + 1 - 0) * 2^(-32) = 1
    let carry_1_0 = (cd_hi_0 + carry_0_0 - FE::zero()) * inv_2_32;
    assert_eq!(carry_1_0, FE::one());
    // Both carries are valid bits
    assert_eq!(carry_1_0 * (FE::one() - carry_1_0), FE::zero());
}

#[test]
fn test_padding_satisfies_constraints() {
    // Padding row: first=0, end=0, mu=0, count=1, address=0, address_incr=[1,0,0,0]
    // count_decr=[0,0,0,0] (count=1 -> count-1=0)
    let inv_2_32 = FE::from(INV_SHIFT_32);
    let one = FE::one();
    let zero = FE::zero();

    // C0-2: IS_BIT for first=0, end=0, mu=0
    assert_eq!(zero * (one - zero), zero);

    // C3: (first + end) * (1 - mu) = (0+0)*(1-0) = 0
    assert_eq!((zero + zero) * (one - zero), zero);

    // C4-5: address + 1 = address_incr
    // addr_lo=0, incr_lo=1 -> carry_0 = (0+1-1)*inv = 0
    let carry_0 = (zero + one - one) * inv_2_32;
    assert_eq!(carry_0, zero);
    assert_eq!(carry_0 * (one - carry_0), zero);
    let carry_1 = (zero + carry_0 - zero) * inv_2_32;
    assert_eq!(carry_1, zero);
    assert_eq!(carry_1 * (one - carry_1), zero);

    // C6-7: count_decr + 1 = count
    // cd_lo=0, count_lo=1 -> carry_0 = (0+1-1)*inv = 0
    let carry_0_sub = (zero + one - one) * inv_2_32;
    assert_eq!(carry_0_sub, zero);
    assert_eq!(carry_0_sub * (one - carry_0_sub), zero);
    let carry_1_sub = (zero + carry_0_sub - zero) * inv_2_32;
    assert_eq!(carry_1_sub, zero);
    assert_eq!(carry_1_sub * (one - carry_1_sub), zero);
}

// =========================================================================
// Edge case tests
// =========================================================================

#[test]
fn test_count_decr_at_zero() {
    // count=0 -> count_decr halfwords all 0xFFFF
    let ops = vec![op(1, 0, 0, 0, true, true, 0)];
    let trace = generate_commit_trace(&ops);
    let r0 = trace.main_table.get_row(0);

    for col in [
        cols::COUNT_DECR_0,
        cols::COUNT_DECR_1,
        cols::COUNT_DECR_2,
        cols::COUNT_DECR_3,
    ] {
        assert_eq!(r0[col], FE::from(0xFFFFu64));
    }
}

#[test]
fn test_address_incr_overflow() {
    // address = 0xFFFF_FFFF_FFFF_FFFF -> address+1 wraps to 0
    let ops = vec![op(1, 0, u64::MAX, 1, true, false, 0xFF)];
    let trace = generate_commit_trace(&ops);
    let r0 = trace.main_table.get_row(0);

    // address = [0xFFFF_FFFF, 0xFFFF_FFFF]
    assert_eq!(r0[cols::ADDRESS_0], FE::from(0xFFFF_FFFFu64));
    assert_eq!(r0[cols::ADDRESS_1], FE::from(0xFFFF_FFFFu64));

    // address_incr = 0 (all halfwords zero)
    assert_eq!(r0[cols::ADDRESS_INCR_0], FE::zero());
    assert_eq!(r0[cols::ADDRESS_INCR_1], FE::zero());
    assert_eq!(r0[cols::ADDRESS_INCR_2], FE::zero());
    assert_eq!(r0[cols::ADDRESS_INCR_3], FE::zero());

    // Verify ADD constraint holds for the wrapped case
    let inv_2_32 = FE::from(INV_SHIFT_32);
    let addr_lo = FE::from(0xFFFF_FFFFu64);
    let addr_hi = FE::from(0xFFFF_FFFFu64);
    // carry_0 = (0xFFFF_FFFF + 1 - 0) * inv = 1
    let carry_0 = (addr_lo + FE::one() - FE::zero()) * inv_2_32;
    assert_eq!(carry_0, FE::one());
    // carry_1 = (0xFFFF_FFFF + 1 - 0) * inv = 1
    let carry_1 = (addr_hi + carry_0 - FE::zero()) * inv_2_32;
    assert_eq!(carry_1, FE::one());
    assert_eq!(carry_0 * (FE::one() - carry_0), FE::zero());
    assert_eq!(carry_1 * (FE::one() - carry_1), FE::zero());
}

#[test]
fn test_word_timestamp() {
    // Timestamps are single 32-bit Words.
    let ts: u64 = 0x0000_0064; // 100
    let ops = vec![op(ts, 0, 0x5000, 1, true, false, 0xAB)];
    let trace = generate_commit_trace(&ops);
    let r0 = trace.main_table.get_row(0);

    assert_eq!(r0[cols::TIMESTAMP], FE::from(ts));
}

#[test]
fn test_minimum_table_size() {
    // Empty ops -> still 4 rows (minimum)
    let trace = generate_commit_trace(&[]);
    assert_eq!(trace.num_rows(), 4);

    // All padding rows
    for row in 0..4 {
        let r = trace.main_table.get_row(row);
        assert_eq!(r[cols::MU], FE::zero());
        assert_eq!(r[cols::COUNT_0], FE::one());
    }
}

#[test]
fn test_address_incr_halfword_carry() {
    // address = 0xFFFF -> address+1 = 0x10000
    // Tests carry propagation across halfwords within the low 32-bit word
    let ops = vec![op(1, 0, 0xFFFF, 1, true, false, 0)];
    let trace = generate_commit_trace(&ops);
    let r0 = trace.main_table.get_row(0);

    // address_incr = 0x10000: h[0]=0x0000, h[1]=0x0001, h[2]=0, h[3]=0
    assert_eq!(r0[cols::ADDRESS_INCR_0], FE::zero());
    assert_eq!(r0[cols::ADDRESS_INCR_1], FE::one());
    assert_eq!(r0[cols::ADDRESS_INCR_2], FE::zero());
    assert_eq!(r0[cols::ADDRESS_INCR_3], FE::zero());
}

#[test]
fn test_bus_interactions_count() {
    use crate::tables::commit::bus_interactions;
    let interactions = bus_interactions();
    assert_eq!(interactions.len(), 18);
}

#[test]
fn test_constraints_count_and_indices() {
    use crate::tables::commit::CommitConstraints;
    use stark::constraints::builder::ConstraintSet;
    let meta = CommitConstraints.meta();
    assert_eq!(meta.len(), 8);
    // Dense, idx-ordered.
    for (i, m) in meta.iter().enumerate() {
        assert_eq!(m.constraint_idx, i);
    }
    // All constraints are degree 2 (unconditional).
    assert_eq!(CommitConstraints.max_degree(), 2);
}
