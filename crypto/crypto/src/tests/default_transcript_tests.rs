use alloc::vec::Vec;
use math::field::{
    element::FieldElement,
    fields::fft_friendly::{
        extensions_goldilocks::Degree3GoldilocksExtensionField,
        u64_goldilocks::GoldilocksField,
    },
};

use crate::fiat_shamir::default_transcript::DefaultTranscript;
use crate::fiat_shamir::is_transcript::IsTranscript;

#[test]
fn basic_challenge() {
    let mut transcript =
        DefaultTranscript::<Degree3GoldilocksExtensionField>::default();

    let point_a: Vec<u8> = vec![0xFF, 0xAB];
    let point_b: Vec<u8> = vec![0xDD, 0x8C, 0x9D];

    transcript.append_bytes(&point_a); // point_a
    transcript.append_bytes(&point_b); // point_a + point_b

    let challenge1 = transcript.sample(); // Hash(point_a  + point_b)

    assert_eq!(
        challenge1,
        [
            0x0c, 0x2b, 0xd8, 0xcf, 0x2d, 0x71, 0xe0, 0x0a, 0xce, 0xa3, 0xbd, 0x5d, 0xc7, 0x9f,
            0x4f, 0x93, 0xed, 0x57, 0x42, 0xd0, 0x23, 0xbd, 0x47, 0xc9, 0x04, 0xc2, 0x67, 0x9d,
            0xbc, 0xfa, 0x7c, 0xa7
        ]
    );

    let point_c: Vec<u8> = vec![0xFF, 0xAB];
    let point_d: Vec<u8> = vec![0xDD, 0x8C, 0x9D];

    transcript.append_bytes(&point_c); // Hash(point_a  + point_b) + point_c
    transcript.append_bytes(&point_d); // Hash(point_a  + point_b) + point_c + point_d

    let challenge2 = transcript.sample(); // Hash(Hash(point_a  + point_b) + point_c + point_d)
    assert_eq!(
        challenge2,
        [
            0x81, 0x61, 0x51, 0xc5, 0x7e, 0xcb, 0x45, 0xd5, 0x17, 0x1a, 0x3c, 0x2e, 0x38, 0x04,
            0x5d, 0xfb, 0x3a, 0x3d, 0x33, 0x8a, 0x22, 0xaf, 0xf8, 0x60, 0x85, 0xb9, 0x54, 0x3f,
            0xf8, 0x32, 0x32, 0xbc
        ]
    );
}

type GoldFE = FieldElement<GoldilocksField>;
type Ext3FE = FieldElement<Degree3GoldilocksExtensionField>;

#[test]
fn degree3_goldilocks_transcript_distinguish_different_fe() {
    let mut transcript_1 = DefaultTranscript::<Degree3GoldilocksExtensionField>::default();
    transcript_1
        .append_field_element(&Ext3FE::new([GoldFE::one(), GoldFE::zero(), GoldFE::zero()]));
    let sample_1 = transcript_1.sample_field_element();

    let mut transcript_2 = DefaultTranscript::<Degree3GoldilocksExtensionField>::default();
    transcript_2
        .append_field_element(&Ext3FE::new([GoldFE::zero(), GoldFE::zero(), GoldFE::one()]));
    let sample_2 = transcript_2.sample_field_element();

    let mut transcript_3 = DefaultTranscript::<Degree3GoldilocksExtensionField>::default();
    transcript_3
        .append_field_element(&Ext3FE::new([GoldFE::one(), GoldFE::zero(), GoldFE::zero()]));
    let sample_3 = transcript_3.sample_field_element();

    assert!(sample_1 != sample_2);
    assert!(sample_1 == sample_3);
}
