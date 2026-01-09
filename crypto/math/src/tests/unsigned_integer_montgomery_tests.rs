use crate::unsigned_integer::{element::U384, montgomery::MontgomeryAlgorithms};

use proptest::prelude::*;

proptest! {
    #[test]
    fn cios_vs_cios_optimized(a in any::<[u64; 6]>(), b in any::<[u64; 6]>()) {
        let x = U384::from_limbs(a);
        let y = U384::from_limbs(b);
        let m = U384::from_hex_unchecked("cdb061954fdd36e5176f50dbdcfd349570a29ce1"); // this is prime
        let mu: u64 = 16085280245840369887; // negative of the inverse of `m` modulo 2^{64}
        assert_eq!(
            MontgomeryAlgorithms::cios(&x, &y, &m, &mu),
            MontgomeryAlgorithms::cios_optimized_for_moduli_with_one_spare_bit(&x, &y, &m, &mu)
        );
    }

    #[test]
    fn cios_vs_sos_square(a in any::<[u64; 6]>()) {
        let x = U384::from_limbs(a);
        let m = U384::from_hex_unchecked("cdb061954fdd36e5176f50dbdcfd349570a29ce1"); // this is prime
        let mu: u64 = 16085280245840369887; // negative of the inverse of `m` modulo 2^{64}
        assert_eq!(
            MontgomeryAlgorithms::cios(&x, &x, &m, &mu),
            MontgomeryAlgorithms::sos_square(&x, &m, &mu)
        );
    }
}
#[test]
fn montgomery_multiplication_works_0() {
    let x = U384::from_u64(11_u64);
    let y = U384::from_u64(10_u64);
    let m = U384::from_u64(23_u64); //
    let mu: u64 = 3208129404123400281; // negative of the inverse of `m` modulo 2^{64}.
    let c = U384::from_u64(13_u64); // x * y * (r^{-1}) % m, where r = 2^{64 * 6} and r^{-1} mod m = 2.
    assert_eq!(MontgomeryAlgorithms::cios(&x, &y, &m, &mu), c);
}

#[test]
fn montgomery_multiplication_works_1() {
    let x = U384::from_hex_unchecked("05ed176deb0e80b4deb7718cdaa075165f149c");
    let y = U384::from_hex_unchecked("5f103b0bd4397d4df560eb559f38353f80eeb6");
    let m = U384::from_hex_unchecked("cdb061954fdd36e5176f50dbdcfd349570a29ce1"); // this is prime
    let mu: u64 = 16085280245840369887; // negative of the inverse of `m` modulo 2^{64}
    let c = U384::from_hex_unchecked("8d65cdee621682815d59f465d2641eea8a1274dc"); // x * y * (r^{-1}) % m, where r = 2^{64 * 6}
    assert_eq!(MontgomeryAlgorithms::cios(&x, &y, &m, &mu), c);
}

#[test]
fn montgomery_multiplication_works_2() {
    let x = U384::from_hex_unchecked("8d65cdee621682815d59f465d2641eea8a1274dc");
    let m = U384::from_hex_unchecked("cdb061954fdd36e5176f50dbdcfd349570a29ce1"); // this is prime
    let r_mod_m = U384::from_hex_unchecked("58dfb0e1b3dd5e674bdcde4f42eb5533b8759d33");
    let mu: u64 = 16085280245840369887; // negative of the inverse of `m` modulo 2^{64}
    let c = U384::from_hex_unchecked("8d65cdee621682815d59f465d2641eea8a1274dc");
    assert_eq!(MontgomeryAlgorithms::cios(&x, &r_mod_m, &m, &mu), c);
}
