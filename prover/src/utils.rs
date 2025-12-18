/// Splits a u32 into 2 limbs of 16 bits each.
pub fn u32_to_2_limbs(x: u32) -> [u32; 2] {
    [x & 0xFFFF, (x >> 16) & 0xFFFF]
}

/// Splits a u32 into 4 limbs of 8 bits each.
pub fn u32_to_4_limbs(x: u32) -> [u32; 4] {
    [
        x & 0xFF,
        (x >> 8) & 0xFF,
        (x >> 16) & 0xFF,
        (x >> 24) & 0xFF,
    ]
}

// TODO: Revisar la logica de como guardar integers in la tabla.
pub fn i32_to_2_limbs(x: i32) -> [u32; 2] {
    let unsigned = x as u32;
    [unsigned & 0xFFFF, (unsigned >> 16) & 0xFFFF]
}

pub fn i32_to_4_limbs(x: i32) -> [u32; 4] {
    let unsigned = x as u32;
    [
        unsigned & 0xFF,
        (unsigned >> 8) & 0xFF,
        (unsigned >> 16) & 0xFF,
        (unsigned >> 24) & 0xFF,
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_u32_to_2_limbs_16() {
        let x = 0xABCD_1234;
        let limbs = u32_to_2_limbs(x);

        assert_eq!(limbs[0], 0x1234);
        assert_eq!(limbs[1], 0xABCD);
    }

    #[test]
    fn test_u32_to_2_limbs_16_zero() {
        let x = 0u32;
        let limbs = u32_to_2_limbs(x);

        assert_eq!(limbs, [0, 0]);
    }

    #[test]
    fn test_u32_to_2_limbs_16_max() {
        let x = u32::MAX;
        let limbs = u32_to_2_limbs(x);

        assert_eq!(limbs, [0xFFFF, 0xFFFF]);
    }

    #[test]
    fn test_u32_to_4_limbs_8() {
        let x = 0xABCD_1234;
        let limbs = u32_to_4_limbs(x);

        assert_eq!(limbs, [0x34, 0x12, 0xCD, 0xAB]);
    }

    #[test]
    fn test_u32_to_4_limbs_8_zero() {
        let x = 0u32;
        let limbs = u32_to_4_limbs(x);

        assert_eq!(limbs, [0, 0, 0, 0]);
    }

    #[test]
    fn test_u32_to_4_limbs_8_max() {
        let x = u32::MAX;
        let limbs = u32_to_4_limbs(x);

        assert_eq!(limbs, [0xFF, 0xFF, 0xFF, 0xFF]);
    }
}
