//! Unit tests for Packing combine logic.

use math::field::element::FieldElement;
use math::field::fields::fft_friendly::babybear::Babybear31PrimeField;

use crate::lookup::{
    AirWithBuses, AuxiliaryTraceBuildData, BusInteraction, NullBoundaryConstraintBuilder, Packing,
};
use crate::proof::options::ProofOptions;
use crate::traits::AIR;

type F = Babybear31PrimeField;
type FE = FieldElement<F>;

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
fn test_dword_wl() {
    // 2 words: [0x12345678, 0xDEADBEEF]
    // Expected: [0x12345678, 0xDEADBEEF] (no combining, just 2 bus elements)
    let words = vec![FE::from(0x12345678u64), FE::from(0xDEADBEEFu64)];
    let combined = Packing::DWordWL.combine(&words);
    assert_eq!(combined.len(), 2);
    assert_eq!(combined[0], FE::from(0x12345678u64));
    assert_eq!(combined[1], FE::from(0xDEADBEEFu64));
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
fn test_air_layout_single_interaction() {
    type E = math::field::fields::fft_friendly::quartic_babybear::Degree4BabyBearExtensionField;

    let interaction = BusInteraction::sender(Some(0), Packing::Direct.columns(&[1, 2, 3]));
    let build_data = AuxiliaryTraceBuildData {
        interactions: vec![interaction],
    };

    let proof_options = ProofOptions::default_test_options();
    let air = AirWithBuses::<F, E, NullBoundaryConstraintBuilder, ()>::new(
        4,
        build_data,
        &proof_options,
        1,
        vec![],
    );

    // 4 main, 2 aux (1 term + 1 accumulated)
    assert_eq!(air.trace_layout(), (4, 2));
}

#[test]
fn test_air_layout_multiple_interactions() {
    type E = math::field::fields::fft_friendly::quartic_babybear::Degree4BabyBearExtensionField;

    let interaction1 = BusInteraction::sender(Some(0), Packing::Direct.columns(&[1, 2]));
    let interaction2 = BusInteraction::sender(Some(0), Packing::Direct.columns(&[3, 4]));
    let build_data = AuxiliaryTraceBuildData {
        interactions: vec![interaction1, interaction2],
    };

    let proof_options = ProofOptions::default_test_options();
    let air = AirWithBuses::<F, E, NullBoundaryConstraintBuilder, ()>::new(
        5,
        build_data,
        &proof_options,
        1,
        vec![],
    );

    // 5 main, 3 aux (2 term + 1 accumulated)
    assert_eq!(air.trace_layout(), (5, 3));
}
