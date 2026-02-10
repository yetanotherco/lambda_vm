use crate::errors::CreationError;
use crate::traits::ByteConversion;
use crate::unsigned_integer::element::{U256, U384, UnsignedInteger};
#[cfg(feature = "proptest")]
use proptest::prelude::*;
#[cfg(feature = "proptest")]
use std::ops::Shr;

#[cfg(feature = "proptest")]
const N_LIMBS: usize = 8;
#[cfg(feature = "proptest")]
type Uint = UnsignedInteger<N_LIMBS>;

#[cfg(feature = "proptest")]
proptest! {
    #[test]
    fn bitand(a in any::<Uint>(), b in any::<Uint>()) {
        let result = a & b;

        for i in 0..N_LIMBS {
            assert_eq!(result.limbs[i], a.limbs[i] & b.limbs[i]);
        }
    }

    #[test]
    fn bitand_assign(a in any::<Uint>(), b in any::<Uint>()) {
        let mut result = a;
        result &= b;

        for i in 0..N_LIMBS {
            assert_eq!(result.limbs[i], a.limbs[i] & b.limbs[i]);
        }
    }

    #[test]
    fn bitor(a in any::<Uint>(), b in any::<Uint>()) {
        let result = a | b;

        for i in 0..N_LIMBS {
            assert_eq!(result.limbs[i], a.limbs[i] | b.limbs[i]);
        }
    }

    #[test]
    fn bitor_assign(a in any::<Uint>(), b in any::<Uint>()) {
        let mut result = a;
        result |= b;

        for i in 0..N_LIMBS {
            assert_eq!(result.limbs[i], a.limbs[i] | b.limbs[i]);
        }
    }

    #[test]
    fn bitxor(a in any::<Uint>(), b in any::<Uint>()) {
        let result = a ^ b;

        for i in 0..N_LIMBS {
            assert_eq!(result.limbs[i], a.limbs[i] ^ b.limbs[i]);
        }
    }

    #[test]
    fn bitxor_assign(a in any::<Uint>(), b in any::<Uint>()) {
        let mut result = a;
        result ^= b;

        for i in 0..N_LIMBS {
            assert_eq!(result.limbs[i], a.limbs[i] ^ b.limbs[i]);
        }
    }

    #[test]
    fn div_rem(a in any::<Uint>(), b in any::<Uint>()) {
        let a = a.shr(256);
        let b = b.shr(256);
        assert_eq!((a * b).div_rem(&b), (a, Uint::from_u64(0)));
    }
}

#[test]
fn construct_new_integer_from_limbs() {
    let a: U384 = UnsignedInteger {
        limbs: [0, 1, 2, 3, 4, 5],
    };
    assert_eq!(U384::from_limbs([0, 1, 2, 3, 4, 5]), a);
}

#[test]
fn construct_new_integer_from_u64_1() {
    let a = U384::from_u64(1_u64);
    assert_eq!(a.limbs, [0, 0, 0, 0, 0, 1]);
}

#[test]
fn construct_new_integer_from_u54_2() {
    let a = U384::from_u64(u64::MAX);
    assert_eq!(a.limbs, [0, 0, 0, 0, 0, u64::MAX]);
}

#[test]
fn construct_new_integer_from_u128_1() {
    let a = U384::from_u128(u128::MAX);
    assert_eq!(a.limbs, [0, 0, 0, 0, u64::MAX, u64::MAX]);
}

#[test]
fn construct_new_integer_from_u128_2() {
    let a = U384::from_u128(276371540478856090688472252609570374439);
    assert_eq!(
        a.limbs,
        [0, 0, 0, 0, 14982131230017065096, 14596400355126379303]
    );
}

#[test]
fn construct_new_integer_from_hex_1() {
    let a = U384::from_hex_unchecked("1");
    assert_eq!(a.limbs, [0, 0, 0, 0, 0, 1]);
}

#[test]
fn construct_new_integer_from_zero_x_1() {
    let a = U384::from_hex_unchecked("0x1");
    assert_eq!(a.limbs, [0, 0, 0, 0, 0, 1]);
}

#[test]
fn construct_new_integer_from_hex_2() {
    let a = U384::from_hex_unchecked("f");
    assert_eq!(a.limbs, [0, 0, 0, 0, 0, 15]);
}

#[test]
fn construct_new_integer_from_hex_3() {
    let a = U384::from_hex_unchecked("10000000000000000");
    assert_eq!(a.limbs, [0, 0, 0, 0, 1, 0]);
}

#[test]
fn construct_new_integer_from_hex_4() {
    let a = U384::from_hex_unchecked("a0000000000000000");
    assert_eq!(a.limbs, [0, 0, 0, 0, 10, 0]);
}

#[test]
fn construct_new_integer_from_hex_5() {
    let a = U384::from_hex_unchecked("ffffffffffffffffff");
    assert_eq!(a.limbs, [0, 0, 0, 0, 255, u64::MAX]);
}

#[test]
fn construct_new_integer_from_hex_6() {
    let a = U384::from_hex_unchecked("eb235f6144d9e91f4b14");
    assert_eq!(a.limbs, [0, 0, 0, 0, 60195, 6872850209053821716]);
}

#[test]
fn construct_new_integer_from_hex_7() {
    let a = U384::from_hex_unchecked("2b20aaa5cf482b239e2897a787faf4660cc95597854beb2");
    assert_eq!(
        a.limbs,
        [
            0,
            0,
            0,
            194229460750598834,
            4171047363999149894,
            6975114134393503410
        ]
    );
}

#[test]
fn construct_new_integer_from_hex_checked_7() {
    let a = U384::from_hex("2b20aaa5cf482b239e2897a787faf4660cc95597854beb2").unwrap();
    assert_eq!(
        a.limbs,
        [
            0,
            0,
            0,
            194229460750598834,
            4171047363999149894,
            6975114134393503410
        ]
    );
}

#[test]
fn construct_new_integer_from_hex_checked_7_with_zero_x() {
    let a = U384::from_hex("0x2b20aaa5cf482b239e2897a787faf4660cc95597854beb2").unwrap();
    assert_eq!(
        a.limbs,
        [
            0,
            0,
            0,
            194229460750598834,
            4171047363999149894,
            6975114134393503410
        ]
    );
}

#[test]
fn construct_new_integer_from_non_hex_errs() {
    assert!(U384::from_hex("0xTEST").is_err());
}

#[test]
fn construct_new_integer_from_empty_string_errs() {
    assert!(U384::from_hex("").is_err());
}

#[test]
fn construct_new_integer_from_hex_checked_8() {
    let a = U384::from_hex("140f5177b90b4f96b61bb8ccb4f298ad2b20aaa5cf482b239e2897a787faf4660cc95597854beb235f6144d9e91f4b14").unwrap();
    assert_eq!(
        a.limbs,
        [
            1445463580056702870,
            13122285128622708909,
            3107671372009581347,
            11396525602857743462,
            921361708038744867,
            6872850209053821716
        ]
    );
}

#[test]
fn construct_new_integer_from_hex_8() {
    let a = U384::from_hex_unchecked(
        "140f5177b90b4f96b61bb8ccb4f298ad2b20aaa5cf482b239e2897a787faf4660cc95597854beb235f6144d9e91f4b14",
    );
    assert_eq!(
        a.limbs,
        [
            1445463580056702870,
            13122285128622708909,
            3107671372009581347,
            11396525602857743462,
            921361708038744867,
            6872850209053821716
        ]
    );
}

#[test]
fn from_hex_with_overflowing_hexstring_should_error() {
    let u256_from_big_string = U256::from_hex(&"f".repeat(65));
    assert!(u256_from_big_string.is_err());
    assert!(u256_from_big_string == Err(CreationError::HexStringIsTooBig));
}

#[test]
fn from_hex_with_non_overflowing_hexstring_should_work() {
    assert_eq!(U256::from_hex(&"0".repeat(64)).unwrap().limbs, [0, 0, 0, 0])
}

#[test]
fn construct_new_integer_from_dec_1() {
    let a = U384::from_dec_str("1").unwrap();
    assert_eq!(a.limbs, [0, 0, 0, 0, 0, 1]);
}

#[test]
fn construct_new_integer_from_dec_2() {
    let a = U384::from_dec_str("15").unwrap();
    assert_eq!(a.limbs, [0, 0, 0, 0, 0, 15]);
}

#[test]
fn construct_new_integer_from_dec_3() {
    let a = U384::from_dec_str("18446744073709551616").unwrap();
    assert_eq!(a.limbs, [0, 0, 0, 0, 1, 0]);
}

#[test]
fn construct_new_integer_from_dec_4() {
    let a = U384::from_dec_str("184467440737095516160").unwrap();
    assert_eq!(a.limbs, [0, 0, 0, 0, 10, 0]);
}

#[test]
fn construct_new_integer_from_dec_5() {
    let a = U384::from_dec_str("4722366482869645213695").unwrap();
    assert_eq!(a.limbs, [0, 0, 0, 0, 255, u64::MAX]);
}

#[test]
fn construct_new_integer_from_dec_6() {
    let a = U384::from_dec_str("1110408632367155513346836").unwrap();
    assert_eq!(a.limbs, [0, 0, 0, 0, 60195, 6872850209053821716]);
}

#[test]
fn construct_new_integer_from_dec_7() {
    let a = U384::from_dec_str("66092860629991288370279803883558073888453977263446474418").unwrap();
    assert_eq!(
        a.limbs,
        [
            0,
            0,
            0,
            194229460750598834,
            4171047363999149894,
            6975114134393503410
        ]
    );
}

#[test]
fn construct_new_integer_from_dec_8() {
    let a = U384::from_dec_str("3087491467896943881295768554872271030441880044814691421073017731442549147034464936390742057449079000462340371991316").unwrap();
    assert_eq!(
        a.limbs,
        [
            1445463580056702870,
            13122285128622708909,
            3107671372009581347,
            11396525602857743462,
            921361708038744867,
            6872850209053821716
        ]
    );
}

#[test]
fn construct_new_integer_from_dec_empty() {
    assert!(U384::from_dec_str("").is_err());
}

#[test]
fn construct_new_integer_from_dec_invalid() {
    assert!(U384::from_dec_str("0xff").is_err());
}

#[test]
fn equality_works_1() {
    let a = U384::from_hex_unchecked("1");
    let b = U384 {
        limbs: [0, 0, 0, 0, 0, 1],
    };
    assert_eq!(a, b);
}
#[test]
fn equality_works_2() {
    let a = U384::from_hex_unchecked("f");
    let b = U384 {
        limbs: [0, 0, 0, 0, 0, 15],
    };
    assert_eq!(a, b);
}

#[test]
fn equality_works_3() {
    let a = U384::from_hex_unchecked("10000000000000000");
    let b = U384 {
        limbs: [0, 0, 0, 0, 1, 0],
    };
    assert_eq!(a, b);
}

#[test]
fn equality_works_4() {
    let a = U384::from_hex_unchecked("a0000000000000000");
    let b = U384 {
        limbs: [0, 0, 0, 0, 10, 0],
    };
    assert_eq!(a, b);
}

#[test]
fn equality_works_5() {
    let a = U384::from_hex_unchecked("ffffffffffffffffff");
    let b = U384 {
        limbs: [0, 0, 0, 0, u8::MAX as u64, u64::MAX],
    };
    assert_eq!(a, b);
}

#[test]
fn equality_works_6() {
    let a = U384::from_hex_unchecked("eb235f6144d9e91f4b14");
    let b = U384 {
        limbs: [0, 0, 0, 0, 60195, 6872850209053821716],
    };
    assert_eq!(a, b);
}

#[test]
fn equality_works_7() {
    let a = U384::from_hex_unchecked("2b20aaa5cf482b239e2897a787faf4660cc95597854beb2");
    let b = U384 {
        limbs: [
            0,
            0,
            0,
            194229460750598834,
            4171047363999149894,
            6975114134393503410,
        ],
    };
    assert_eq!(a, b);
}

#[test]
fn equality_works_8() {
    let a = U384::from_hex_unchecked(
        "140f5177b90b4f96b61bb8ccb4f298ad2b20aaa5cf482b239e2897a787faf4660cc95597854beb235f6144d9e91f4b14",
    );
    let b = U384 {
        limbs: [
            1445463580056702870,
            13122285128622708909,
            3107671372009581347,
            11396525602857743462,
            921361708038744867,
            6872850209053821716,
        ],
    };
    assert_eq!(a, b);
}

#[test]
fn equality_works_9() {
    let a = U384::from_hex_unchecked("fffffff");
    let b = U384::from_hex_unchecked("fefffff");
    assert_ne!(a, b);
}

#[test]
fn equality_works_10() {
    let a = U384::from_hex_unchecked("ffff000000000000");
    let b = U384::from_hex_unchecked("ffff000000100000");
    assert_ne!(a, b);
}

#[test]
fn const_ne_works_1() {
    let a = U384::from_hex_unchecked("ffff000000000000");
    let b = U384::from_hex_unchecked("ffff000000100000");
    assert!(U384::const_ne(&a, &b));
}

#[test]
fn const_ne_works_2() {
    let a = U384::from_hex_unchecked(
        "140f5177b90b4f96b61bb8ccb4f298ad2b20aaa5cf482b239e2897a787faf4660cc95597854beb235f6144d9e91f4b14",
    );
    let b = U384 {
        limbs: [
            1445463580056702870,
            13122285128622708909,
            3107671372009581347,
            11396525602857743462,
            921361708038744867,
            6872850209053821716,
        ],
    };
    assert!(!U384::const_ne(&a, &b));
}

#[test]
fn double_two_384_bit_integers() {
    let a = U384::from_u64(2);
    let b = U384::from_u64(5);
    let c = U384::from_u64(7);
    assert_eq!(U384::double(&a).0, a + a);
    assert_eq!(U384::double(&b).0, b + b);
    assert_eq!(U384::double(&c).0, c + c);
}

#[test]
fn add_two_384_bit_integers_1() {
    let a = U384::from_u64(2);
    let b = U384::from_u64(5);
    let c = U384::from_u64(7);
    assert_eq!(a + b, c);
}

#[test]
fn add_two_384_bit_integers_2() {
    let a = U384::from_u64(334);
    let b = U384::from_u64(666);
    let c = U384::from_u64(1000);
    assert_eq!(a + b, c);
}

#[test]
fn add_two_384_bit_integers_3() {
    let a = U384::from_hex_unchecked("ffffffffffffffff");
    let b = U384::from_hex_unchecked("1");
    let c = U384::from_hex_unchecked("10000000000000000");
    assert_eq!(a + b, c);
}

#[test]
fn add_two_384_bit_integers_4() {
    let a = U384::from_hex_unchecked("b58e1e0b66");
    let b = U384::from_hex_unchecked("55469d9619");
    let c = U384::from_hex_unchecked("10ad4bba17f");
    assert_eq!(a + b, c);
}

#[test]
fn add_two_384_bit_integers_5() {
    let a = U384::from_hex_unchecked("e8dff25cb6160f7705221da6f");
    let b = U384::from_hex_unchecked("ab879169b5f80dc8a7969f0b0");
    let c = U384::from_hex_unchecked("1946783c66c0e1d3facb8bcb1f");
    assert_eq!(a + b, c);
}

#[test]
fn add_two_384_bit_integers_6() {
    let a = U384::from_hex_unchecked("9adf291af3a64d59e14e7b440c850508014c551ed5");
    let b = U384::from_hex_unchecked("e7948474bce907f0feaf7e5d741a8cd2f6d1fb9448");
    let c = U384::from_hex_unchecked("18273ad8fb08f554adffdf9a1809f91daf81e50b31d");
    assert_eq!(a + b, c);
}

#[test]
fn add_two_384_bit_integers_7() {
    let a = U384::from_hex_unchecked(
        "f866aef803c92bf02e85c7fad0eccb4881c59825e499fa22f98e1a8fefed4cd9a03647cd3cc84",
    );
    let b = U384::from_hex_unchecked(
        "9b4000dccf01a010e196154a1b998408f949d734389626ba97cb3331ee87e01dd5badc58f41b2",
    );
    let c = U384::from_hex_unchecked(
        "193a6afd4d2cacc01101bdd44ec864f517b0f6f5a1d3020dd91594dc1de752cf775f1242630e36",
    );
    assert_eq!(a + b, c);
}

#[test]
fn add_two_384_bit_integers_8() {
    let a = U384::from_hex_unchecked(
        "07df9c74fa9d5aafa74a87dbbf93215659d8a3e1706d4b06de9512284802580eb36ae12ea59f90db5b1799d0970a42e",
    );
    let b = U384::from_hex_unchecked(
        "d515e54973f0643a6a9957579c1f84020a6a91d5d5f27b75401c7538d2c9ea9cafff44a2c606877d46c49a3433cc85e",
    );
    let c = U384::from_hex_unchecked(
        "dcf581be6e8dbeea11e3df335bb2a558644335b7465fc67c1eb187611acc42ab636a25d16ba61858a1dc3404cad6c8c",
    );
    assert_eq!(a + b, c);
}

#[test]
fn add_two_384_bit_integers_9() {
    let a = U384::from_hex_unchecked(
        "92977527a0f8ba00d18c1b2f1900d965d4a70e5f5f54468ffb2d4d41519385f24b078a0e7d0281d5ad0c36724dc4233",
    );
    let b = U384::from_hex_unchecked(
        "46facf9953a9494822bf18836ffd7e55c48b30aa81e17fa1ace0b473015307e4622b8bd6fa68ef654796a183abde842",
    );
    let c = U384::from_hex_unchecked(
        "d99244c0f4a20348f44b33b288fe57bb99323f09e135c631a80e01b452e68dd6ad3315e5776b713af4a2d7f5f9a2a75",
    );
    assert_eq!(a + b, c);
}

#[test]
fn add_two_384_bit_integers_10() {
    let a = U384::from_hex_unchecked(
        "07df9c74fa9d5aafa74a87dbbf93215659d8a3e1706d4b06de9512284802580eb36ae12ea59f90db5b1799d0970a42e",
    );
    let b = U384::from_hex_unchecked(
        "d515e54973f0643a6a9957579c1f84020a6a91d5d5f27b75401c7538d2c9ea9cafff44a2c606877d46c49a3433cc85e",
    );
    let c_expected = U384::from_hex_unchecked(
        "dcf581be6e8dbeea11e3df335bb2a558644335b7465fc67c1eb187611acc42ab636a25d16ba61858a1dc3404cad6c8c",
    );
    let (c, overflow) = U384::add(&a, &b);
    assert_eq!(c, c_expected);
    assert!(!overflow);
}

#[test]
fn add_two_384_bit_integers_11() {
    let a = U384::from_hex_unchecked(
        "92977527a0f8ba00d18c1b2f1900d965d4a70e5f5f54468ffb2d4d41519385f24b078a0e7d0281d5ad0c36724dc4233",
    );
    let b = U384::from_hex_unchecked(
        "46facf9953a9494822bf18836ffd7e55c48b30aa81e17fa1ace0b473015307e4622b8bd6fa68ef654796a183abde842",
    );
    let c_expected = U384::from_hex_unchecked(
        "d99244c0f4a20348f44b33b288fe57bb99323f09e135c631a80e01b452e68dd6ad3315e5776b713af4a2d7f5f9a2a75",
    );
    let (c, overflow) = U384::add(&a, &b);
    assert_eq!(c, c_expected);
    assert!(!overflow);
}

#[test]
fn add_two_384_bit_integers_12_with_overflow() {
    let a = U384::from_hex_unchecked(
        "b07bc844363dd56467d9ebdd5929e9bb34a8e2577db77df6cf8f2ac45bd3d0bc2fc3078d265fe761af51d6aec5b59428",
    );
    let b = U384::from_hex_unchecked(
        "cbbc474761bb7995ff54e25fa5d30295604fe3545d0cde405e72d8c0acebb119e9158131679b6c34483a3dafb49deeea",
    );
    let c_expected = U384::from_hex_unchecked(
        "7c380f8b97f94efa672ece3cfefcec5094f8c5abdac45c372e02038508bf81d618d888be8dfb5395f78c145e7a538312",
    );
    let (c, overflow) = U384::add(&a, &b);
    assert_eq!(c, c_expected);
    assert!(overflow);
}

#[test]
fn double_384_bit_integer_12_with_overflow() {
    let a = U384::from_hex_unchecked(
        "b07bc844363dd56467d9ebdd5929e9bb34a8e2577db77df6cf8f2ac45bd3d0bc2fc3078d265fe761af51d6aec5b59428",
    );
    assert_eq!(U384::double(&a), U384::add(&a, &a));
}

#[test]
fn sub_two_384_bit_integers_1() {
    let a = U384::from_u64(2);
    let b = U384::from_u64(5);
    let c = U384::from_u64(7);
    assert_eq!(c - a, b);
}

#[test]
fn sub_two_384_bit_integers_2() {
    let a = U384::from_u64(334);
    let b = U384::from_u64(666);
    let c = U384::from_u64(1000);
    assert_eq!(c - a, b);
}

#[test]
fn sub_two_384_bit_integers_3() {
    let a = U384::from_hex_unchecked("ffffffffffffffff");
    let b = U384::from_hex_unchecked("1");
    let c = U384::from_hex_unchecked("10000000000000000");
    assert_eq!(c - a, b);
}

#[test]
fn sub_two_384_bit_integers_4() {
    let a = U384::from_hex_unchecked("b58e1e0b66");
    let b = U384::from_hex_unchecked("55469d9619");
    let c = U384::from_hex_unchecked("10ad4bba17f");
    assert_eq!(c - a, b);
}

#[test]
fn sub_two_384_bit_integers_5() {
    let a = U384::from_hex_unchecked("e8dff25cb6160f7705221da6f");
    let b = U384::from_hex_unchecked("ab879169b5f80dc8a7969f0b0");
    let c = U384::from_hex_unchecked("1946783c66c0e1d3facb8bcb1f");
    assert_eq!(c - a, b);
}

#[test]
fn sub_two_384_bit_integers_6() {
    let a = U384::from_hex_unchecked("9adf291af3a64d59e14e7b440c850508014c551ed5");
    let b = U384::from_hex_unchecked("e7948474bce907f0feaf7e5d741a8cd2f6d1fb9448");
    let c = U384::from_hex_unchecked("18273ad8fb08f554adffdf9a1809f91daf81e50b31d");
    assert_eq!(c - a, b);
}

#[test]
fn sub_two_384_bit_integers_7() {
    let a = U384::from_hex_unchecked(
        "f866aef803c92bf02e85c7fad0eccb4881c59825e499fa22f98e1a8fefed4cd9a03647cd3cc84",
    );
    let b = U384::from_hex_unchecked(
        "9b4000dccf01a010e196154a1b998408f949d734389626ba97cb3331ee87e01dd5badc58f41b2",
    );
    let c = U384::from_hex_unchecked(
        "193a6afd4d2cacc01101bdd44ec864f517b0f6f5a1d3020dd91594dc1de752cf775f1242630e36",
    );
    assert_eq!(c - a, b);
}

#[test]
fn sub_two_384_bit_integers_8() {
    let a = U384::from_hex_unchecked(
        "07df9c74fa9d5aafa74a87dbbf93215659d8a3e1706d4b06de9512284802580eb36ae12ea59f90db5b1799d0970a42e",
    );
    let b = U384::from_hex_unchecked(
        "d515e54973f0643a6a9957579c1f84020a6a91d5d5f27b75401c7538d2c9ea9cafff44a2c606877d46c49a3433cc85e",
    );
    let c = U384::from_hex_unchecked(
        "dcf581be6e8dbeea11e3df335bb2a558644335b7465fc67c1eb187611acc42ab636a25d16ba61858a1dc3404cad6c8c",
    );
    assert_eq!(c - a, b);
}

#[test]
fn sub_two_384_bit_integers_9() {
    let a = U384::from_hex_unchecked(
        "92977527a0f8ba00d18c1b2f1900d965d4a70e5f5f54468ffb2d4d41519385f24b078a0e7d0281d5ad0c36724dc4233",
    );
    let b = U384::from_hex_unchecked(
        "46facf9953a9494822bf18836ffd7e55c48b30aa81e17fa1ace0b473015307e4622b8bd6fa68ef654796a183abde842",
    );
    let c = U384::from_hex_unchecked(
        "d99244c0f4a20348f44b33b288fe57bb99323f09e135c631a80e01b452e68dd6ad3315e5776b713af4a2d7f5f9a2a75",
    );
    assert_eq!(c - a, b);
}

#[test]
fn sub_two_384_bit_integers_11_without_overflow() {
    let a = U384::from_u64(334);
    let b_expected = U384::from_u64(666);
    let c = U384::from_u64(1000);
    let (b, underflow) = U384::sub(&c, &a);
    assert!(!underflow);
    assert_eq!(b_expected, b);
}

#[test]
fn sub_two_384_bit_integers_11_with_overflow() {
    let a = U384::from_u64(334);
    let b_expected = U384::from_hex_unchecked(
        "fffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffd66",
    );
    let c = U384::from_u64(1000);
    let (b, underflow) = U384::sub(&a, &c);
    assert!(underflow);
    assert_eq!(b_expected, b);
}

#[test]
fn partial_order_works() {
    assert!(U384::from_u64(10) <= U384::from_u64(10));
    assert!(U384::from_u64(1) < U384::from_u64(2));
    assert!(U384::from_u64(2) >= U384::from_u64(1));

    assert!(U384::from_u64(10) >= U384::from_u64(10));
    assert!(U384::from_u64(2) > U384::from_u64(1));
    assert!(U384::from_u64(1) <= U384::from_u64(2));

    let a = U384::from_hex_unchecked(
        "92977527a0f8ba00d18c1b2f1900d965d4a70e5f5f54468ffb2d4d41519385f24b078a0e7d0281d5ad0c36724dc4233",
    );
    let c = U384::from_hex_unchecked(
        "d99244c0f4a20348f44b33b288fe57bb99323f09e135c631a80e01b452e68dd6ad3315e5776b713af4a2d7f5f9a2a75",
    );

    assert!(a <= a);
    assert!(a >= a);
    assert!(a >= a);
    assert!(a <= a);
    assert!(a < (a + U384::from_u64(1)));
    assert!(a <= (a + U384::from_u64(1)));
    assert!(a + U384::from_u64(1) > a);
    assert!((a + U384::from_u64(1) >= a));
    assert!(a <= c);
    assert!(a < c);
    assert!(a < c);
    assert!(a <= c);
    assert!(c > a);
    assert!(c >= a);
    assert!(c >= a);
    assert!(c > a);
    assert!(a < c);
}

#[test]
fn mul_two_384_bit_integers_works_1() {
    let a = U384::from_u64(3);
    let b = U384::from_u64(8);
    let c = U384::from_u64(3 * 8);
    assert_eq!(a * b, c);
}

#[test]
fn mul_two_384_bit_integers_works_2() {
    let a = U384::from_hex_unchecked("6131d99f840b3b0");
    let b = U384::from_hex_unchecked("6f5c466db398f43");
    let c = U384::from_hex_unchecked("2a47a603a77f871dfbb937af7e5710");
    assert_eq!(a * b, c);
}

#[test]
fn mul_two_384_bit_integers_works_3() {
    let a = U384::from_hex_unchecked("84a6add5db9e095b2e0f6b40eff8ee");
    let b = U384::from_hex_unchecked("2347db918f725461bec2d5c57");
    let c = U384::from_hex_unchecked("124805c476c9462adc0df6c88495d4253f5c38033afc18d78d920e2");
    assert_eq!(a * b, c);
}

#[test]
fn mul_two_384_bit_integers_works_4() {
    let a = U384::from_hex_unchecked("04050753dd7c0b06c404633016f87040");
    let b = U384::from_hex_unchecked("dc3830be041b3b4476445fcad3dac0f6f3a53e4ba12da");
    let c = U384::from_hex_unchecked(
        "375342999dab7f52f4010c4abc2e18b55218015931a55d6053ac39e86e2a47d6b1cb95f41680",
    );
    assert_eq!(a * b, c);
}

#[test]
fn mul_two_384_bit_integers_works_5() {
    let a = U384::from_hex_unchecked(
        "7ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff8",
    );
    let b = U384::from_hex_unchecked("2");
    let c_expected = U384::from_hex_unchecked(
        "fffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff0",
    );
    assert_eq!(a * b, c_expected);
}

#[test]
#[should_panic]
fn mul_two_384_bit_integers_works_6() {
    let a = U384::from_hex_unchecked(
        "800000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000",
    );
    let b = U384::from_hex_unchecked("2");
    let _c = a * b;
}

#[test]
fn mul_two_384_bit_integers_works_7_hi_lo() {
    let a = U384::from_hex_unchecked("04050753dd7c0b06c404633016f87040");
    let b = U384::from_hex_unchecked("dc3830be041b3b4476445fcad3dac0f6f3a53e4ba12da");
    let hi_expected = U384::from_hex_unchecked("0");
    let lo_expected = U384::from_hex_unchecked(
        "375342999dab7f52f4010c4abc2e18b55218015931a55d6053ac39e86e2a47d6b1cb95f41680",
    );
    let (hi, lo) = U384::mul(&a, &b);
    assert_eq!(hi, hi_expected);
    assert_eq!(lo, lo_expected);
}

#[test]
fn mul_two_384_bit_integers_works_8_hi_lo() {
    let a = U384::from_hex_unchecked(
        "5e2d939b602a50911232731d04fe6f40c05f97da0602307099fb991f9b414e2d52bef130349ec18db1a0215ea6caf76",
    );
    let b = U384::from_hex_unchecked(
        "3f3ad1611ab58212f92a2484e9560935b9ac4615fe61cfed1a4861e193a74d20c94f9f88d8b2cc089543c3f699969d9",
    );
    let hi_expected = U384::from_hex_unchecked(
        "1742daad9c7861dd3499e7ece65467e337937b27e20d641b225bfe00323d33ed62715654eadc092b057a5f19f2ad6c",
    );
    let lo_expected = U384::from_hex_unchecked(
        "9969c0417b9304d9c16b046c860447d3533999e16710d2e90a44959a168816c015ffb44b987e8cbb82bd46b08d9e2106",
    );
    let (hi, lo) = U384::mul(&a, &b);
    assert_eq!(hi, hi_expected);
    assert_eq!(lo, lo_expected);
}

#[test]
fn mul_two_384_bit_integers_works_9_hi_lo() {
    let a = U384::from_hex_unchecked(
        "800000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000",
    );
    let b = U384::from_hex_unchecked("2");
    let hi_expected = U384::from_hex_unchecked("1");
    let lo_expected = U384::from_hex_unchecked("0");
    let (hi, lo) = U384::mul(&a, &b);
    assert_eq!(hi, hi_expected);
    assert_eq!(lo, lo_expected);
}

#[test]
fn shift_left_on_384_bit_integer_works_1() {
    let a = U384::from_hex_unchecked("1");
    let b = U384::from_hex_unchecked("10");
    assert_eq!(a << 4, b);
}

#[test]
fn shift_left_on_384_bit_integer_works_2() {
    let a = U384::from_u64(1);
    let b = U384::from_u128(1_u128 << 64);
    assert_eq!(a << 64, b);
}

#[test]
fn shift_left_on_384_bit_integer_works_3() {
    let a = U384::from_hex_unchecked("10");
    let b = U384::from_hex_unchecked("1000");
    assert_eq!(a << 8, b);
}

#[test]
fn shift_left_on_384_bit_integer_works_4() {
    let a = U384::from_hex_unchecked("e45542992b6844553f3cb1c5ac33e7fa5");
    let b = U384::from_hex_unchecked("391550a64ada11154fcf2c716b0cf9fe940");
    assert_eq!(a << 6, b);
}

#[test]
fn shift_left_on_384_bit_integer_works_5() {
    let a = U384::from_hex_unchecked(
        "03303f4d6c2d1caf0c24a6b0239b679a8390aa99bead76bc0093b1bc1a8101f5ce",
    );
    let b = U384::from_hex_unchecked(
        "6607e9ad85a395e18494d604736cf35072155337d5aed7801276378350203eb9c0000000000000000000000000000000",
    );
    assert_eq!(a << 125, b);
}

#[test]
fn shift_left_on_384_bit_integer_works_6() {
    let a = U384::from_hex_unchecked("762e8968bc392ed786ab132f0b5b0cacd385dd51de3a");
    let b = U384::from_hex_unchecked(
        "762e8968bc392ed786ab132f0b5b0cacd385dd51de3a00000000000000000000000000000000",
    );
    assert_eq!(a << (64 * 2), b);
}

#[test]
fn shift_left_on_384_bit_integer_works_7() {
    let a = U384::from_hex_unchecked("90823e0bd707f");
    let b =
        U384::from_hex_unchecked("90823e0bd707f000000000000000000000000000000000000000000000000");
    assert_eq!(a << (64 * 3), b);
}

#[test]
fn shift_right_on_384_bit_integer_works_1() {
    let a = U384::from_hex_unchecked("1");
    let b = U384::from_hex_unchecked("10");
    assert_eq!(b >> 4, a);
}

#[test]
fn shift_right_on_384_bit_integer_works_2() {
    let a = U384::from_hex_unchecked("10");
    let b = U384::from_hex_unchecked("1000");
    assert_eq!(b >> 8, a);
}

#[test]
fn shift_right_on_384_bit_integer_works_3() {
    let a = U384::from_hex_unchecked("e45542992b6844553f3cb1c5ac33e7fa5");
    let b = U384::from_hex_unchecked("391550a64ada11154fcf2c716b0cf9fe940");
    assert_eq!(b >> 6, a);
}

#[test]
fn shift_right_on_384_bit_integer_works_4() {
    let a = U384::from_hex_unchecked(
        "03303f4d6c2d1caf0c24a6b0239b679a8390aa99bead76bc0093b1bc1a8101f5ce",
    );
    let b = U384::from_hex_unchecked(
        "6607e9ad85a395e18494d604736cf35072155337d5aed7801276378350203eb9c0000000000000000000000000000000",
    );
    assert_eq!(b >> 125, a);
}

#[test]
fn shift_right_on_384_bit_integer_works_5() {
    let a = U384::from_hex_unchecked(
        "ba6ab46f9a9a2f20e4061b67ce4d8c3da98091cf990d7b14ef47ffe27370abbdeb6a3ce9f9cbf5df1b2430114c8558eb",
    );
    let b = U384::from_hex_unchecked("174d568df35345e41c80c36cf9c9b187b5301239f321af629de8fffc4e6");
    assert_eq!(a >> 151, b);
}

#[test]
fn shift_right_on_384_bit_integer_works_6() {
    let a = U384::from_hex_unchecked(
        "076c075d2f65e39b9ecdde8bf6f8c94241962ce0f557b7739673200c777152eb7e772ad35",
    );
    let b = U384::from_hex_unchecked("ed80eba5ecbc7373d9bbd17edf19284832c59c1eaaf6ee7");
    assert_eq!(a >> 99, b);
}

#[test]
fn shift_right_on_384_bit_integer_works_7() {
    let a = U384::from_hex_unchecked(
        "6a9ce35d8940a5ebd29604ce9a182ade76f03f7e9965760b84a8cfd1d3dd2e612669fe000e58b2af688fd90",
    );
    let b = U384::from_hex_unchecked("6a9ce35d8940a5ebd29604ce9a182ade76f03f7");
    assert_eq!(a >> (64 * 3), b);
}

#[test]
fn shift_right_on_384_bit_integer_works_8() {
    let a = U384::from_hex_unchecked(
        "5322c128ec84081b6c376c108ebd7fd36bbd44f71ee5e6ad6bcb3dd1c5265bd7db75c90b2665a0826d17600f0e9",
    );
    let b = U384::from_hex_unchecked("5322c128ec84081b6c376c108ebd7fd36bbd44f71ee5e6ad6bcb3dd1c52");
    assert_eq!(a >> (64 * 2), b);
}

#[test]
#[cfg(feature = "alloc")]
fn to_be_bytes_works() {
    let number = U384::from_u64(1);
    let expected_bytes = [
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1,
    ];

    assert_eq!(number.to_bytes_be(), expected_bytes);
}

#[test]
#[cfg(feature = "alloc")]
fn to_le_bytes_works() {
    let number = U384::from_u64(1);
    let expected_bytes = [
        1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    ];

    assert_eq!(number.to_bytes_le(), expected_bytes);
}

#[test]
fn from_bytes_be_works() {
    let bytes = [
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1,
    ];
    let expected_number = U384::from_u64(1);

    assert_eq!(U384::from_bytes_be(&bytes).unwrap(), expected_number);
}

#[test]
fn from_bytes_le_works() {
    let bytes = [
        1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    ];
    let expected_number = U384::from_u64(1);

    assert_eq!(U384::from_bytes_le(&bytes).unwrap(), expected_number);
}

#[test]
fn from_bytes_be_works_with_extra_data() {
    let bytes = [
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    ];
    let expected_number = U384::from_u64(0);

    assert_eq!(U384::from_bytes_be(&bytes).unwrap(), expected_number);
}

#[test]
#[should_panic]
fn from_bytes_be_errs_with_less_data() {
    let bytes = [0, 0, 0, 0, 0];
    U384::from_bytes_be(&bytes).unwrap();
}

#[test]
#[should_panic]
fn from_bytes_le_errs_with_less_data() {
    let bytes = [0, 0, 0, 0, 0];
    U384::from_bytes_le(&bytes).unwrap();
}

#[test]
fn from_bytes_le_works_with_extra_data() {
    let bytes = [
        1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1,
    ];
    let expected_number = U384::from_u64(1);

    assert_eq!(U384::from_bytes_le(&bytes).unwrap(), expected_number);
}

#[test]
fn test_square_0() {
    let a = U384::from_hex_unchecked(
        "362e35606447fb568704026c25da7a304bc7bd0aea36a61d77d4151395078cfa332b9d4928a60721eece725bbc81e158",
    );
    let (hi, lo) = U384::square(&a);
    assert_eq!(
        lo,
        U384::from_hex_unchecked(
            "11724caeb10c4bce5319097d74aed2246e2942b56b7365b5b2f8ceb3bb847db4828862043299d798577996e210bce40"
        )
    );
    assert_eq!(
        hi,
        U384::from_hex_unchecked(
            "b7786dbe41375b7ff64dbdc65152ef7d3fdbf499485e26486201cdbfb71b5673c77eb355a1274d08cbfbc1a4cdfdfad"
        )
    );
}

#[test]
fn test_square_1() {
    let a = U384::from_limbs([0, 0, 0, 0, 0, u64::MAX]);
    let (hi, lo) = U384::square(&a);
    assert_eq!(
        lo,
        U384::from_hex_unchecked("fffffffffffffffe0000000000000001")
    );
    assert_eq!(hi, U384::from_hex_unchecked("0"));
}

#[test]
fn test_square_2() {
    let a = U384::from_limbs([0, 0, 0, 0, u64::MAX, 0]);
    let (hi, lo) = U384::square(&a);
    assert_eq!(
        lo,
        U384::from_hex_unchecked(
            "fffffffffffffffe000000000000000100000000000000000000000000000000"
        )
    );
    assert_eq!(hi, U384::from_hex_unchecked("0"));
}

#[test]
fn test_square_3() {
    let a = U384::from_limbs([0, 0, 0, u64::MAX, 0, 0]);
    let (hi, lo) = U384::square(&a);
    assert_eq!(
        lo,
        U384::from_hex_unchecked(
            "fffffffffffffffe00000000000000010000000000000000000000000000000000000000000000000000000000000000"
        )
    );
    assert_eq!(hi, U384::from_hex_unchecked("0"));
}

#[test]
fn test_square_4() {
    let a = U384::from_limbs([0, 0, u64::MAX, 0, 0, 0]);
    let (hi, lo) = U384::square(&a);
    assert_eq!(lo, U384::from_hex_unchecked("0"));
    assert_eq!(
        hi,
        U384::from_hex_unchecked("fffffffffffffffe0000000000000001")
    );
}

#[test]
fn test_square_5() {
    let a = U384::from_limbs([0, 0, u64::MAX, u64::MAX, u64::MAX, u64::MAX]);
    let (hi, lo) = U384::square(&a);
    assert_eq!(
        lo,
        U384::from_hex_unchecked(
            "fffffffffffffffffffffffffffffffe0000000000000000000000000000000000000000000000000000000000000001"
        )
    );
    assert_eq!(
        hi,
        U384::from_hex_unchecked("ffffffffffffffffffffffffffffffff")
    );
}

#[test]
fn test_square_6() {
    let a = U384::from_limbs([0, u64::MAX, u64::MAX, u64::MAX, u64::MAX, u64::MAX]);
    let (hi, lo) = U384::square(&a);
    assert_eq!(
        lo,
        U384::from_hex_unchecked(
            "fffffffffffffffe00000000000000000000000000000000000000000000000000000000000000000000000000000001"
        )
    );
    assert_eq!(
        hi,
        U384::from_hex_unchecked(
            "ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff"
        )
    );
}

#[test]
fn test_square_7() {
    let a = U384::from_limbs([u64::MAX, u64::MAX, u64::MAX, u64::MAX, u64::MAX, u64::MAX]);
    let (hi, lo) = U384::square(&a);
    assert_eq!(lo, U384::from_hex_unchecked("1"));
    assert_eq!(
        hi,
        U384::from_hex_unchecked(
            "fffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffe"
        )
    );
}
