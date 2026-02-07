use math::field::{
    element::FieldElement, fields::fft_friendly::u64_goldilocks::GoldilocksField,
};

type FE = FieldElement<GoldilocksField>;

pub fn u32_to_2_limbs(x: u32) -> [FE; 2] {
    [(x & 0xFFFF) as u64, ((x >> 16) & 0xFFFF) as u64].map(FE::new)
}

pub fn u32_to_4_limbs(x: u32) -> [FE; 4] {
    [
        (x & 0xFF) as u64,
        ((x >> 8) & 0xFF) as u64,
        ((x >> 16) & 0xFF) as u64,
        ((x >> 24) & 0xFF) as u64,
    ]
    .map(FE::new)
}

pub fn i32_to_2_limbs(x: i32) -> [FE; 2] {
    let unsigned = x as u32;
    [(unsigned & 0xFFFF) as u64, ((unsigned >> 16) & 0xFFFF) as u64].map(FE::new)
}

pub fn i32_to_4_limbs(x: i32) -> [FE; 4] {
    let unsigned = x as u32;
    [
        (unsigned & 0xFF) as u64,
        ((unsigned >> 8) & 0xFF) as u64,
        ((unsigned >> 16) & 0xFF) as u64,
        ((unsigned >> 24) & 0xFF) as u64,
    ]
    .map(FE::new)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_u32_to_2_limbs_16() {
        let x = 0xABCD_1234;
        let limbs = u32_to_2_limbs(x);
        assert_eq!(limbs[0], FE::new(0x1234));
        assert_eq!(limbs[1], FE::new(0xABCD));
    }

    #[test]
    fn test_u32_to_2_limbs_16_zero() {
        let x = 0u32;
        let limbs = u32_to_2_limbs(x);
        assert_eq!(limbs, [FE::new(0), FE::new(0)]);
    }

    #[test]
    fn test_u32_to_2_limbs_16_max() {
        let x = u32::MAX;
        let limbs = u32_to_2_limbs(x);
        assert_eq!(limbs, [FE::new(0xFFFF), FE::new(0xFFFF)]);
    }

    #[test]
    fn test_u32_to_4_limbs_8() {
        let x = 0xABCD_1234;
        let limbs = u32_to_4_limbs(x);
        assert_eq!(
            limbs,
            [FE::new(0x34), FE::new(0x12), FE::new(0xCD), FE::new(0xAB)]
        );
    }

    #[test]
    fn test_u32_to_4_limbs_8_zero() {
        let x = 0u32;
        let limbs = u32_to_4_limbs(x);
        assert_eq!(limbs, [FE::new(0), FE::new(0), FE::new(0), FE::new(0)]);
    }

    #[test]
    fn test_u32_to_4_limbs_8_max() {
        let x = u32::MAX;
        let limbs = u32_to_4_limbs(x);
        assert_eq!(
            limbs,
            [FE::new(0xFF), FE::new(0xFF), FE::new(0xFF), FE::new(0xFF)]
        );
    }
}
