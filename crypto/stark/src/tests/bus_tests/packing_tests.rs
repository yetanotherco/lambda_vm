//! Unit tests for Packing combine logic.

use crate::constraints::builder::EmptyConstraints;
use math::field::element::FieldElement;
use math::field::goldilocks::GoldilocksField;

use crate::lookup::{
    AirWithBuses, AuxiliaryTraceBuildData, BusInteraction, Multiplicity,
    NullBoundaryConstraintBuilder, Packing,
};
use crate::proof::options::ProofOptions;
use crate::traits::AIR;

type F = GoldilocksField;
type FE = FieldElement<F>;

/// Bus ID for packing tests (single bus)
const TEST_BUS: u64 = 0;

#[test]
fn test_direct() {
    // Direct: 1 column -> 1 bus element, no combining
    let values = vec![FE::from(42u64)];
    let combined = Packing::Direct.combine(&values);
    assert_eq!(combined.len(), 1);
    assert_eq!(combined[0], FE::from(42u64));
}

#[test]
fn test_word4l() {
    // 4 bytes: [0x12, 0x34, 0x56, 0x78]
    // Expected: 0x78563412 (little-endian)
    let bytes = vec![
        FE::from(0x12u64),
        FE::from(0x34u64),
        FE::from(0x56u64),
        FE::from(0x78u64),
    ];
    let combined = Packing::Word4L.combine(&bytes);
    assert_eq!(combined.len(), 1);
    assert_eq!(combined[0], FE::from(0x78563412u64));
}

#[test]
fn test_word2l() {
    // 2 halves: [0x1234, 0x5678]
    // Expected: 0x56781234 (little-endian)
    let halves = vec![FE::from(0x1234u64), FE::from(0x5678u64)];
    let combined = Packing::Word2L.combine(&halves);
    assert_eq!(combined.len(), 1);
    assert_eq!(combined[0], FE::from(0x56781234u64));
}

#[test]
fn test_dword_hl() {
    // 4 halves: [0x1234, 0x5678, 0x9ABC, 0xDEF0]
    // Expected: [0x56781234, 0xDEF09ABC]
    let halves = vec![
        FE::from(0x1234u64),
        FE::from(0x5678u64),
        FE::from(0x9ABCu64),
        FE::from(0xDEF0u64),
    ];
    let combined = Packing::DWordHL.combine(&halves);
    assert_eq!(combined.len(), 2);
    assert_eq!(combined[0], FE::from(0x56781234u64));
    assert_eq!(combined[1], FE::from(0xDEF09ABCu64));
}

#[test]
fn test_quad_hl() {
    // 8 halves: [0x1111, 0x2222, 0x3333, 0x4444, 0x5555, 0x6666, 0x7777, 0x8888]
    // Expected: 4 words via 4× Word2L
    // [0x22221111, 0x44443333, 0x66665555, 0x88887777]
    let halves = vec![
        FE::from(0x1111u64),
        FE::from(0x2222u64),
        FE::from(0x3333u64),
        FE::from(0x4444u64),
        FE::from(0x5555u64),
        FE::from(0x6666u64),
        FE::from(0x7777u64),
        FE::from(0x8888u64),
    ];
    let combined = Packing::QuadHL.combine(&halves);
    assert_eq!(combined.len(), 4);
    assert_eq!(combined[0], FE::from(0x22221111u64));
    assert_eq!(combined[1], FE::from(0x44443333u64));
    assert_eq!(combined[2], FE::from(0x66665555u64));
    assert_eq!(combined[3], FE::from(0x88887777u64));
}

#[test]
fn test_dword_hhw() {
    // [Word, Half, Half] where Word is LSB
    // columns: [0xAABBCCDD, 0x1234, 0x5678]
    // Expected: [0xAABBCCDD, 0x56781234]
    let cols = vec![
        FE::from(0xAABBCCDDu64),
        FE::from(0x1234u64),
        FE::from(0x5678u64),
    ];
    let combined = Packing::DWordHHW.combine(&cols);
    assert_eq!(combined.len(), 2);
    assert_eq!(combined[0], FE::from(0xAABBCCDDu64));
    assert_eq!(combined[1], FE::from(0x56781234u64));
}

#[test]
fn test_dword_whh() {
    // [Half, Half, Word] where Word is MSB
    // columns: [0x1234, 0x5678, 0xAABBCCDD]
    // Expected: [0x56781234, 0xAABBCCDD]
    let cols = vec![
        FE::from(0x1234u64),
        FE::from(0x5678u64),
        FE::from(0xAABBCCDDu64),
    ];
    let combined = Packing::DWordWHH.combine(&cols);
    assert_eq!(combined.len(), 2);
    assert_eq!(combined[0], FE::from(0x56781234u64));
    assert_eq!(combined[1], FE::from(0xAABBCCDDu64));
}

#[test]
fn test_dword_bl() {
    // 8 bytes → 2 words (2× Word4L)
    let bytes = vec![
        FE::from(0x11u64),
        FE::from(0x22u64),
        FE::from(0x33u64),
        FE::from(0x44u64),
        FE::from(0x55u64),
        FE::from(0x66u64),
        FE::from(0x77u64),
        FE::from(0x88u64),
    ];
    let combined = Packing::DWordBL.combine(&bytes);
    assert_eq!(combined.len(), 2);
    // First word: 0x11 + 0x22*2^8 + 0x33*2^16 + 0x44*2^24 = 0x44332211
    assert_eq!(combined[0], FE::from(0x44332211u64));
    // Second word: 0x55 + 0x66*2^8 + 0x77*2^16 + 0x88*2^24 = 0x88776655
    assert_eq!(combined[1], FE::from(0x88776655u64));
}

#[test]
fn test_dword_wl() {
    // 2 words → 2 bus elements (no combining, 2× Direct)
    // columns: [0x11223344, 0x55667788]
    // Expected: [0x11223344, 0x55667788] (pass-through)
    let words = vec![FE::from(0x11223344u64), FE::from(0x55667788u64)];
    let combined = Packing::DWordWL.combine(&words);
    assert_eq!(combined.len(), 2);
    assert_eq!(combined[0], FE::from(0x11223344u64));
    assert_eq!(combined[1], FE::from(0x55667788u64));
}

// =============================================================================
// Compound delegation tests
// =============================================================================
// These tests verify that compound packings produce identical results
// to manually applying the primitives they're built from.

#[test]
fn test_dword_wl_equals_two_direct() {
    let words = vec![FE::from(0x11223344u64), FE::from(0x55667788u64)];

    // Compound
    let compound_result = Packing::DWordWL.combine(&words);

    // Manual: 2× Direct
    let mut manual_result = Packing::Direct.combine(&words[0..1]);
    manual_result.extend(Packing::Direct.combine(&words[1..2]));

    assert_eq!(compound_result, manual_result);
}

#[test]
fn test_dword_hl_equals_two_word2l() {
    let halves = vec![
        FE::from(0x1234u64),
        FE::from(0x5678u64),
        FE::from(0x9ABCu64),
        FE::from(0xDEF0u64),
    ];

    // Compound
    let compound_result = Packing::DWordHL.combine(&halves);

    // Manual: 2× Word2L
    let mut manual_result = Packing::Word2L.combine(&halves[0..2]);
    manual_result.extend(Packing::Word2L.combine(&halves[2..4]));

    assert_eq!(compound_result, manual_result);
}

#[test]
fn test_dword_bl_equals_two_word4l() {
    let bytes: Vec<FE> = (1u64..=8).map(FE::from).collect();

    // Compound
    let compound_result = Packing::DWordBL.combine(&bytes);

    // Manual: 2× Word4L
    let mut manual_result = Packing::Word4L.combine(&bytes[0..4]);
    manual_result.extend(Packing::Word4L.combine(&bytes[4..8]));

    assert_eq!(compound_result, manual_result);
}

#[test]
fn test_dword_hhw_equals_direct_plus_word2l() {
    let cols = vec![
        FE::from(0xAABBCCDDu64),
        FE::from(0x1234u64),
        FE::from(0x5678u64),
    ];

    // Compound
    let compound_result = Packing::DWordHHW.combine(&cols);

    // Manual: Direct + Word2L
    let mut manual_result = Packing::Direct.combine(&cols[0..1]);
    manual_result.extend(Packing::Word2L.combine(&cols[1..3]));

    assert_eq!(compound_result, manual_result);
}

#[test]
fn test_dword_whh_equals_word2l_plus_direct() {
    let cols = vec![
        FE::from(0x1234u64),
        FE::from(0x5678u64),
        FE::from(0xAABBCCDDu64),
    ];

    // Compound
    let compound_result = Packing::DWordWHH.combine(&cols);

    // Manual: Word2L + Direct
    let mut manual_result = Packing::Word2L.combine(&cols[0..2]);
    manual_result.extend(Packing::Direct.combine(&cols[2..3]));

    assert_eq!(compound_result, manual_result);
}

#[test]
fn test_quad_hl_equals_four_word2l() {
    let halves: Vec<FE> = (1u64..=8).map(|x| FE::from(x * 0x1111)).collect();

    // Compound
    let compound_result = Packing::QuadHL.combine(&halves);

    // Manual: 4× Word2L
    let mut manual_result = Packing::Word2L.combine(&halves[0..2]);
    manual_result.extend(Packing::Word2L.combine(&halves[2..4]));
    manual_result.extend(Packing::Word2L.combine(&halves[4..6]));
    manual_result.extend(Packing::Word2L.combine(&halves[6..8]));

    assert_eq!(compound_result, manual_result);
}

#[test]
fn test_quad_wl() {
    // 4 words → 4 bus elements (no combining, 4× Direct)
    // columns: [0x11111111, 0x22222222, 0x33333333, 0x44444444]
    // Expected: [0x11111111, 0x22222222, 0x33333333, 0x44444444] (pass-through)
    let words = vec![
        FE::from(0x11111111u64),
        FE::from(0x22222222u64),
        FE::from(0x33333333u64),
        FE::from(0x44444444u64),
    ];
    let combined = Packing::QuadWL.combine(&words);
    assert_eq!(combined.len(), 4);
    assert_eq!(combined[0], FE::from(0x11111111u64));
    assert_eq!(combined[1], FE::from(0x22222222u64));
    assert_eq!(combined[2], FE::from(0x33333333u64));
    assert_eq!(combined[3], FE::from(0x44444444u64));
}

#[test]
fn test_quad_wl_equals_four_direct() {
    let words = vec![
        FE::from(0xAABBCCDDu64),
        FE::from(0x11223344u64),
        FE::from(0x55667788u64),
        FE::from(0x99AABBCCu64),
    ];

    // Compound
    let compound_result = Packing::QuadWL.combine(&words);

    // Manual: 4× Direct
    let mut manual_result = Packing::Direct.combine(&words[0..1]);
    manual_result.extend(Packing::Direct.combine(&words[1..2]));
    manual_result.extend(Packing::Direct.combine(&words[2..3]));
    manual_result.extend(Packing::Direct.combine(&words[3..4]));

    assert_eq!(compound_result, manual_result);
}

// =============================================================================
// AIR layout tests
// =============================================================================

#[test]
fn test_air_layout_single_interaction() {
    type E = math::field::extensions_goldilocks::Degree3GoldilocksExtensionField;

    let interaction = BusInteraction::sender(
        TEST_BUS,
        Multiplicity::Column(0),
        Packing::Direct.columns(&[1, 2, 3]),
    );
    let build_data = AuxiliaryTraceBuildData {
        interactions: vec![interaction],
    };

    let proof_options = ProofOptions::default_test_options();
    let air = AirWithBuses::<F, E, NullBoundaryConstraintBuilder, (), _>::new(
        4,
        build_data,
        &proof_options,
        1,
        EmptyConstraints,
    );

    // 4 main, 1 aux (0 committed pairs + 1 accumulated with 1 absorbed)
    assert_eq!(air.trace_layout(), (4, 1));
}

#[test]
fn test_air_layout_multiple_interactions() {
    type E = math::field::extensions_goldilocks::Degree3GoldilocksExtensionField;

    let interaction1 = BusInteraction::sender(
        TEST_BUS,
        Multiplicity::Column(0),
        Packing::Direct.columns(&[1, 2]),
    );
    let interaction2 = BusInteraction::sender(
        TEST_BUS,
        Multiplicity::Column(0),
        Packing::Direct.columns(&[3, 4]),
    );
    let build_data = AuxiliaryTraceBuildData {
        interactions: vec![interaction1, interaction2],
    };

    let proof_options = ProofOptions::default_test_options();
    let air = AirWithBuses::<F, E, NullBoundaryConstraintBuilder, (), _>::new(
        5,
        build_data,
        &proof_options,
        1,
        EmptyConstraints,
    );

    // 5 main, 1 aux (0 committed pairs + 1 accumulated with 2 absorbed)
    assert_eq!(air.trace_layout(), (5, 1));
}
