use crate::errors::CreationError;
use crate::traits::ByteConversion;
use crate::unsigned_integer::element::{U256, UnsignedInteger};
#[cfg(feature = "proptest")]
use proptest::prelude::*;
#[cfg(feature = "proptest")]
use std::ops::Shr;

#[cfg(feature = "proptest")]
const N_LIMBS: usize = 4;
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
        let a = a.shr(128);
        let b = b.shr(128);
        assert_eq!((a * b).div_rem(&b), (a, Uint::from_u64(0)));
    }
}

#[test]
fn construct_new_integer_from_limbs() {
    let a: U256 = UnsignedInteger {
        limbs: [0, 1, 2, 3],
    };
    assert_eq!(U256::from_limbs([0, 1, 2, 3]), a);
}

#[test]
fn construct_new_integer_from_u64_1() {
    let a = U256::from_u64(1_u64);
    assert_eq!(a.limbs, [0, 0, 0, 1]);
}

#[test]
fn construct_new_integer_from_u64_2() {
    let a = U256::from_u64(u64::MAX);
    assert_eq!(a.limbs, [0, 0, 0, u64::MAX]);
}

#[test]
fn construct_new_integer_from_u128_1() {
    let a = U256::from_u128(u128::MAX);
    assert_eq!(a.limbs, [0, 0, u64::MAX, u64::MAX]);
}

#[test]
fn construct_new_integer_from_u128_4() {
    let a = U256::from_u128(276371540478856090688472252609570374439);
    assert_eq!(a.limbs, [0, 0, 14982131230017065096, 14596400355126379303]);
}

#[test]
fn construct_new_integer_from_hex_1() {
    let a = U256::from_hex_unchecked("1");
    assert_eq!(a.limbs, [0, 0, 0, 1]);
}

#[test]
fn construct_new_integer_from_hex_2() {
    let a = U256::from_hex_unchecked("f");
    assert_eq!(a.limbs, [0, 0, 0, 15]);
}

#[test]
fn construct_new_integer_from_hex_3() {
    let a = U256::from_hex_unchecked("10000000000000000");
    assert_eq!(a.limbs, [0, 0, 1, 0]);
}

#[test]
fn construct_new_integer_from_hex_4() {
    let a = U256::from_hex_unchecked("a0000000000000000");
    assert_eq!(a.limbs, [0, 0, 10, 0]);
}

#[test]
fn construct_new_integer_from_hex_5() {
    let a = U256::from_hex_unchecked("ffffffffffffffffff");
    assert_eq!(a.limbs, [0, 0, 255, u64::MAX]);
}

#[test]
fn construct_new_integer_from_hex_6() {
    let a = U256::from_hex_unchecked("eb235f6144d9e91f4b14");
    assert_eq!(a.limbs, [0, 0, 60195, 6872850209053821716]);
}

#[test]
fn construct_new_integer_from_hex_7() {
    let a = U256::from_hex_unchecked("2b20aaa5cf482b239e2897a787faf4660cc95597854beb2");
    assert_eq!(
        a.limbs,
        [
            0,
            194229460750598834,
            4171047363999149894,
            6975114134393503410
        ]
    );
}

#[test]
fn construct_new_integer_from_hex_8() {
    let a = U256::from_hex_unchecked(
        "2B20AAA5CF482B239E2897A787FAF4660CC95597854BEB235F6144D9E91F4B14",
    );
    assert_eq!(
        a.limbs,
        [
            3107671372009581347,
            11396525602857743462,
            921361708038744867,
            6872850209053821716
        ]
    );
}

#[test]
fn construct_new_integer_from_dec_1() {
    let a = U256::from_dec_str("1").unwrap();
    assert_eq!(a.limbs, [0, 0, 0, 1]);
}

#[test]
fn construct_integer_from_invalid_hex_returns_error() {
    assert_eq!(U256::from_hex("0xaO"), Err(CreationError::InvalidHexString));
    assert_eq!(U256::from_hex("0xOa"), Err(CreationError::InvalidHexString));
    assert_eq!(U256::from_hex("0xm"), Err(CreationError::InvalidHexString));
}

#[test]
fn construct_new_integer_from_dec_2() {
    let a = U256::from_dec_str("15").unwrap();
    assert_eq!(a.limbs, [0, 0, 0, 15]);
}

#[test]
fn construct_new_integer_from_dec_3() {
    let a = U256::from_dec_str("18446744073709551616").unwrap();
    assert_eq!(a.limbs, [0, 0, 1, 0]);
}

#[test]
fn construct_new_integer_from_dec_4() {
    let a = U256::from_dec_str("184467440737095516160").unwrap();
    assert_eq!(a.limbs, [0, 0, 10, 0]);
}

#[test]
fn construct_new_integer_from_dec_5() {
    let a = U256::from_dec_str("4722366482869645213695").unwrap();
    assert_eq!(a.limbs, [0, 0, 255, u64::MAX]);
}

#[test]
fn construct_new_integer_from_dec_6() {
    let a = U256::from_dec_str("1110408632367155513346836").unwrap();
    assert_eq!(a.limbs, [0, 0, 60195, 6872850209053821716]);
}

#[test]
fn construct_new_integer_from_dec_7() {
    let a = U256::from_dec_str("66092860629991288370279803883558073888453977263446474418").unwrap();
    assert_eq!(
        a.limbs,
        [
            0,
            194229460750598834,
            4171047363999149894,
            6975114134393503410
        ]
    );
}

#[test]
fn construct_new_integer_from_dec_8() {
    let a = U256::from_dec_str(
        "19507169362252850253634654373914901165934018806002526957372506333098895428372",
    )
    .unwrap();
    assert_eq!(
        a.limbs,
        [
            3107671372009581347,
            11396525602857743462,
            921361708038744867,
            6872850209053821716
        ]
    );
}

#[test]
fn construct_new_integer_from_dec_empty() {
    assert!(U256::from_dec_str("").is_err());
}

#[test]
fn construct_new_integer_from_dec_invalid() {
    assert!(U256::from_dec_str("0xff").is_err());
}

#[test]
fn equality_works_1() {
    let a = U256::from_hex_unchecked("1");
    let b = U256 {
        limbs: [0, 0, 0, 1],
    };
    assert_eq!(a, b);
}
#[test]
fn equality_works_2() {
    let a = U256::from_hex_unchecked("f");
    let b = U256 {
        limbs: [0, 0, 0, 15],
    };
    assert_eq!(a, b);
}

#[test]
fn equality_works_3() {
    let a = U256::from_hex_unchecked("10000000000000000");
    let b = U256 {
        limbs: [0, 0, 1, 0],
    };
    assert_eq!(a, b);
}

#[test]
fn equality_works_4() {
    let a = U256::from_hex_unchecked("a0000000000000000");
    let b = U256 {
        limbs: [0, 0, 10, 0],
    };
    assert_eq!(a, b);
}

#[test]
fn equality_works_5() {
    let a = U256::from_hex_unchecked("ffffffffffffffffff");
    let b = U256 {
        limbs: [0, 0, u8::MAX as u64, u64::MAX],
    };
    assert_eq!(a, b);
}

#[test]
fn equality_works_6() {
    let a = U256::from_hex_unchecked("eb235f6144d9e91f4b14");
    let b = U256 {
        limbs: [0, 0, 60195, 6872850209053821716],
    };
    assert_eq!(a, b);
}

#[test]
fn equality_works_7() {
    let a = U256::from_hex_unchecked("2b20aaa5cf482b239e2897a787faf4660cc95597854beb2");
    let b = U256 {
        limbs: [
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
    let a = U256::from_hex_unchecked(
        "2B20AAA5CF482B239E2897A787FAF4660CC95597854BEB235F6144D9E91F4B14",
    );
    let b = U256 {
        limbs: [
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
    let a = U256::from_hex_unchecked("fffffff");
    let b = U256::from_hex_unchecked("fefffff");
    assert_ne!(a, b);
}

#[test]
fn equality_works_10() {
    let a = U256::from_hex_unchecked("ffff000000000000");
    let b = U256::from_hex_unchecked("ffff000000100000");
    assert_ne!(a, b);
}

#[test]
fn double_256_bit_integer_1() {
    let a = U256::from_u64(2);
    let b = U256::from_u64(5);
    let c = U256::from_u64(7);
    assert_eq!(U256::double(&a).0, a + a);
    assert_eq!(U256::double(&b).0, b + b);
    assert_eq!(U256::double(&c).0, c + c);
}

#[test]
fn add_two_256_bit_integers_1() {
    let a = U256::from_u64(2);
    let b = U256::from_u64(5);
    let c = U256::from_u64(7);
    assert_eq!(a + b, c);
}

#[test]
fn add_two_256_bit_integers_2() {
    let a = U256::from_u64(334);
    let b = U256::from_u64(666);
    let c = U256::from_u64(1000);
    assert_eq!(a + b, c);
}

#[test]
fn add_two_256_bit_integers_3() {
    let a = U256::from_hex_unchecked("ffffffffffffffff");
    let b = U256::from_hex_unchecked("1");
    let c = U256::from_hex_unchecked("10000000000000000");
    assert_eq!(a + b, c);
}

#[test]
fn add_two_256_bit_integers_4() {
    let a = U256::from_hex_unchecked("b58e1e0b66");
    let b = U256::from_hex_unchecked("55469d9619");
    let c = U256::from_hex_unchecked("10ad4bba17f");
    assert_eq!(a + b, c);
}

#[test]
fn add_two_256_bit_integers_5() {
    let a = U256::from_hex_unchecked("e8dff25cb6160f7705221da6f");
    let b = U256::from_hex_unchecked("ab879169b5f80dc8a7969f0b0");
    let c = U256::from_hex_unchecked("1946783c66c0e1d3facb8bcb1f");
    assert_eq!(a + b, c);
}

#[test]
fn add_two_256_bit_integers_6() {
    let a = U256::from_hex_unchecked("9adf291af3a64d59e14e7b440c850508014c551ed5");
    let b = U256::from_hex_unchecked("e7948474bce907f0feaf7e5d741a8cd2f6d1fb9448");
    let c = U256::from_hex_unchecked("18273ad8fb08f554adffdf9a1809f91daf81e50b31d");
    assert_eq!(a + b, c);
}

#[test]
fn add_two_256_bit_integers_7() {
    let a = U256::from_hex_unchecked(
        "10d3bc05496380cfe27bf5d97ddb99ac95eb5ecfbd3907eadf877a4c2dfa05f6",
    );
    let b = U256::from_hex_unchecked(
        "0866aef803c92bf02e85c7fad0eccb4881c59825e499fa22f98e1a8fefed4cd9",
    );
    let c = U256::from_hex_unchecked(
        "193a6afd4d2cacc01101bdd44ec864f517b0f6f5a1d3020dd91594dc1de752cf",
    );
    assert_eq!(a + b, c);
}

#[test]
fn add_two_256_bit_integers_8() {
    let a = U256::from_hex_unchecked(
        "07df9c74fa9d5aafa74a87dbbf93215659d8a3e1706d4b06de9512284802580f",
    );
    let b = U256::from_hex_unchecked(
        "d515e54973f0643a6a9957579c1f84020a6a91d5d5f27b75401c7538d2c9ea9c",
    );
    let c = U256::from_hex_unchecked(
        "dcf581be6e8dbeea11e3df335bb2a558644335b7465fc67c1eb187611acc42ab",
    );
    assert_eq!(a + b, c);
}

#[test]
fn add_two_256_bit_integers_9() {
    let a = U256::from_hex_unchecked(
        "92977527a0f8ba00d18c1b2f1900d965d4a70e5f5f54468ffb2d4d41519385f2",
    );
    let b = U256::from_hex_unchecked(
        "46facf9953a9494822bf18836ffd7e55c48b30aa81e17fa1ace0b473015307e4",
    );
    let c = U256::from_hex_unchecked(
        "d99244c0f4a20348f44b33b288fe57bb99323f09e135c631a80e01b452e68dd6",
    );
    assert_eq!(a + b, c);
}

#[test]
fn add_two_256_bit_integers_10() {
    let a = U256::from_hex_unchecked(
        "07df9c74fa9d5aafa74a87dbbf93215659d8a3e1706d4b06de9512284802580f",
    );
    let b = U256::from_hex_unchecked(
        "d515e54973f0643a6a9957579c1f84020a6a91d5d5f27b75401c7538d2c9ea9c",
    );
    let c_expected = U256::from_hex_unchecked(
        "dcf581be6e8dbeea11e3df335bb2a558644335b7465fc67c1eb187611acc42ab",
    );
    let (c, overflow) = U256::add(&a, &b);
    assert_eq!(c, c_expected);
    assert!(!overflow);
}

#[test]
fn add_two_256_bit_integers_11() {
    let a = U256::from_hex_unchecked(
        "92977527a0f8ba00d18c1b2f1900d965d4a70e5f5f54468ffb2d4d41519385f2",
    );
    let b = U256::from_hex_unchecked(
        "46facf9953a9494822bf18836ffd7e55c48b30aa81e17fa1ace0b473015307e4",
    );
    let c_expected = U256::from_hex_unchecked(
        "d99244c0f4a20348f44b33b288fe57bb99323f09e135c631a80e01b452e68dd6",
    );
    let (c, overflow) = U256::add(&a, &b);
    assert_eq!(c, c_expected);
    assert!(!overflow);
}

#[test]
fn add_two_256_bit_integers_12_with_overflow() {
    let a = U256::from_hex_unchecked(
        "b07bc844363dd56467d9ebdd5929e9bb34a8e2577db77df6cf8f2ac45bd3d0bc",
    );
    let b = U256::from_hex_unchecked(
        "cbbc474761bb7995ff54e25fa5d30295604fe3545d0cde405e72d8c0acebb119",
    );
    let c_expected = U256::from_hex_unchecked(
        "7c380f8b97f94efa672ece3cfefcec5094f8c5abdac45c372e02038508bf81d5",
    );
    let (c, overflow) = U256::add(&a, &b);
    assert_eq!(c, c_expected);
    assert!(overflow);
}

#[test]
fn double_256_bit_integer_12_with_overflow() {
    let a = U256::from_hex_unchecked(
        "b07bc844363dd56467d9ebdd5929e9bb34a8e2577db77df6cf8f2ac45bd3d0bc",
    );
    let b = U256::from_hex_unchecked(
        "cbbc474761bb7995ff54e25fa5d30295604fe3545d0cde405e72d8c0acebb119",
    );
    assert_eq!(U256::double(&a), U256::add(&a, &a));
    assert_eq!(U256::double(&b), U256::add(&b, &b));
}

#[test]
fn sub_two_256_bit_integers_1() {
    let a = U256::from_u64(2);
    let b = U256::from_u64(5);
    let c = U256::from_u64(7);
    assert_eq!(c - a, b);
}

#[test]
fn sub_two_256_bit_integers_2() {
    let a = U256::from_u64(334);
    let b = U256::from_u64(666);
    let c = U256::from_u64(1000);
    assert_eq!(c - a, b);
}

#[test]
fn sub_two_256_bit_integers_3() {
    let a = U256::from_hex_unchecked("ffffffffffffffff");
    let b = U256::from_hex_unchecked("1");
    let c = U256::from_hex_unchecked("10000000000000000");
    assert_eq!(c - a, b);
}

#[test]
fn sub_two_256_bit_integers_4() {
    let a = U256::from_hex_unchecked("b58e1e0b66");
    let b = U256::from_hex_unchecked("55469d9619");
    let c = U256::from_hex_unchecked("10ad4bba17f");
    assert_eq!(c - a, b);
}

#[test]
fn sub_two_256_bit_integers_5() {
    let a = U256::from_hex_unchecked("e8dff25cb6160f7705221da6f");
    let b = U256::from_hex_unchecked("ab879169b5f80dc8a7969f0b0");
    let c = U256::from_hex_unchecked("1946783c66c0e1d3facb8bcb1f");
    assert_eq!(c - a, b);
}

#[test]
fn sub_two_256_bit_integers_6() {
    let a = U256::from_hex_unchecked("9adf291af3a64d59e14e7b440c850508014c551ed5");
    let b = U256::from_hex_unchecked("e7948474bce907f0feaf7e5d741a8cd2f6d1fb9448");
    let c = U256::from_hex_unchecked("18273ad8fb08f554adffdf9a1809f91daf81e50b31d");
    assert_eq!(c - a, b);
}

#[test]
fn sub_two_256_bit_integers_7() {
    let a = U256::from_hex_unchecked(
        "9b4000dccf01a010e196154a1b998408f949d734389626ba97cb3331ee87e01d",
    );
    let b = U256::from_hex_unchecked(
        "5d26ae1b34c78bdf4cefb2b0b553473f887bc0f1ac03d36861c2e75e01656cbc",
    );
    let c = U256::from_hex_unchecked(
        "f866aef803c92bf02e85c7fad0eccb4881c59825e499fa22f98e1a8fefed4cd9",
    );
    assert_eq!(c - a, b);
}

#[test]
fn sub_two_256_bit_integers_8() {
    let a = U256::from_hex_unchecked(
        "07df9c74fa9d5aafa74a87dbbf93215659d8a3e1706d4b06de9512284802580e",
    );
    let b = U256::from_hex_unchecked(
        "d515e54973f0643a6a9957579c1f84020a6a91d5d5f27b75401c7538d2c9ea9d",
    );
    let c = U256::from_hex_unchecked(
        "dcf581be6e8dbeea11e3df335bb2a558644335b7465fc67c1eb187611acc42ab",
    );
    assert_eq!(c - a, b);
}

#[test]
fn sub_two_256_bit_integers_9() {
    let a = U256::from_hex_unchecked(
        "92977527a0f8ba00d18c1b2f1900d965d4a70e5f5f54468ffb2d4d41519385f2",
    );
    let b = U256::from_hex_unchecked(
        "46facf9953a9494822bf18836ffd7e55c48b30aa81e17fa1ace0b473015307e4",
    );
    let c = U256::from_hex_unchecked(
        "d99244c0f4a20348f44b33b288fe57bb99323f09e135c631a80e01b452e68dd6",
    );
    assert_eq!(c - a, b);
}

#[test]
fn sub_two_256_bit_integers_11_without_overflow() {
    let a = U256::from_u64(334);
    let b_expected = U256::from_u64(666);
    let c = U256::from_u64(1000);
    let (b, overflow) = U256::sub(&c, &a);
    assert!(!overflow);
    assert_eq!(b_expected, b);
}

#[test]
fn sub_two_256_bit_integers_11_with_overflow() {
    let a = U256::from_u64(334);
    let b_expected = U256::from_hex_unchecked(
        "fffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffd66",
    );
    let c = U256::from_u64(1000);
    let (b, overflow) = U256::sub(&a, &c);
    assert!(overflow);
    assert_eq!(b_expected, b);
}

#[test]
fn const_le_works() {
    let a = U256::from_u64(334);
    let b = U256::from_u128(333);
    assert!(U256::const_le(&b, &a));
    assert!(U256::const_le(&a, &a));
    assert!(!U256::const_le(&a, &b));
}

#[test]
fn partial_order_works() {
    assert!(U256::from_u64(10) <= U256::from_u64(10));
    assert!(U256::from_u64(1) < U256::from_u64(2));
    assert!(U256::from_u64(2) >= U256::from_u64(1));

    assert!(U256::from_u64(10) >= U256::from_u64(10));
    assert!(U256::from_u64(2) > U256::from_u64(1));
    assert!(U256::from_u64(1) <= U256::from_u64(2));

    let a = U256::from_hex_unchecked(
        "5d4a70e5f5f54468ffb2d4d41519385f24b078a0e7d0281d5ad0c36724dc4233",
    );
    let c = U256::from_hex_unchecked(
        "b99323f09e135c631a80e01b452e68dd6ad3315e5776b713af4a2d7f5f9a2a75",
    );

    assert!(a <= a);
    assert!(a >= a);
    assert!(a >= a);
    assert!(a <= a);
    assert!(a < (a + U256::from_u64(1)));
    assert!(a <= (a + U256::from_u64(1)));
    assert!(a + U256::from_u64(1) > a);
    assert!((a + U256::from_u64(1) >= a));
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
fn mul_two_256_bit_integers_works_1() {
    let a = U256::from_u64(3);
    let b = U256::from_u64(8);
    let c = U256::from_u64(3 * 8);
    assert_eq!(a * b, c);
}

#[test]
fn mul_two_256_bit_integers_works_2() {
    let a = U256::from_hex_unchecked("6131d99f840b3b0");
    let b = U256::from_hex_unchecked("6f5c466db398f43");
    let c = U256::from_hex_unchecked("2a47a603a77f871dfbb937af7e5710");
    assert_eq!(a * b, c);
}

#[test]
fn mul_two_256_bit_integers_works_3() {
    let a = U256::from_hex_unchecked("84a6add5db9e095b2e0f6b40eff8ee");
    let b = U256::from_hex_unchecked("2347db918f725461bec2d5c57");
    let c = U256::from_hex_unchecked("124805c476c9462adc0df6c88495d4253f5c38033afc18d78d920e2");
    assert_eq!(a * b, c);
}

#[test]
fn mul_two_256_bit_integers_works_4() {
    let a = U256::from_hex_unchecked("15bf61fcf53a3f0ae1e8e555d");
    let b = U256::from_hex_unchecked("cbbc474761bb7995ff54e25fa5d5d0cde405e9f");
    let c_expected = U256::from_hex_unchecked(
        "114ec14db0c80d30b7dcb9c45948ef04cc149e612cb544f447b146553aff2ac3",
    );
    assert_eq!(a * b, c_expected);
}

#[test]
fn mul_two_256_bit_integers_works_5_hi_lo() {
    let a = U256::from_hex_unchecked(
        "8e2d939b602a50911232731d04fe6f40c05f97da0602307099fb991f9b414e2d",
    );
    let b = U256::from_hex_unchecked(
        "7f3ad1611ab58212f92a2484e9560935b9ac4615fe61cfed1a4861e193a74d20",
    );
    let hi_expected = U256::from_hex_unchecked(
        "46A946D6A984FE6507DE6B8D1354256D7A7BAE4283404733BDC876A264BCE5EE",
    );
    let lo_expected = U256::from_hex_unchecked(
        "43F24263F10930EBE3EA0307466C19B13B9C7DBA6B3F7604B7F32FB0E3084EA0",
    );
    let (hi, lo) = U256::mul(&a, &b);
    assert_eq!(hi, hi_expected);
    assert_eq!(lo, lo_expected);
}

#[test]
fn shift_left_on_256_bit_integer_works_1() {
    let a = U256::from_hex_unchecked("1");
    let b = U256::from_hex_unchecked("10");
    assert_eq!(a << 4, b);
}

#[test]
fn shift_left_on_256_bit_integer_works_2() {
    let a = U256::from_u64(1);
    let b = U256::from_u128(1_u128 << 64);
    assert_eq!(a << 64, b);
}

#[test]
fn shift_left_on_256_bit_integer_works_3() {
    let a = U256::from_hex_unchecked("10");
    let b = U256::from_hex_unchecked("1000");
    assert_eq!(&a << 8, b);
}

#[test]
fn shift_left_on_256_bit_integer_works_4() {
    let a = U256::from_hex_unchecked("e45542992b6844553f3cb1c5ac33e7fa5");
    let b = U256::from_hex_unchecked("391550a64ada11154fcf2c716b0cf9fe940");
    assert_eq!(a << 6, b);
}

#[test]
fn shift_left_on_256_bit_integer_works_5() {
    let a = U256::from_hex_unchecked("a8390aa99bead76bc0093b1bc1a8101f5ce");
    let b = U256::from_hex_unchecked(
        "72155337d5aed7801276378350203eb9c0000000000000000000000000000000",
    );
    assert_eq!(&a << 125, b);
}

#[test]
fn shift_left_on_256_bit_integer_works_6() {
    let a = U256::from_hex_unchecked("2ed786ab132f0b5b0cacd385dd51de3a");
    let b = U256::from_hex_unchecked(
        "2ed786ab132f0b5b0cacd385dd51de3a00000000000000000000000000000000",
    );
    assert_eq!(a << (64 * 2), b);
}

#[test]
fn shift_left_on_256_bit_integer_works_7() {
    let a = U256::from_hex_unchecked("90823e0bd707f");
    let b =
        U256::from_hex_unchecked("90823e0bd707f000000000000000000000000000000000000000000000000");
    assert_eq!(a << (64 * 3), b);
}

#[test]
fn shift_right_on_256_bit_integer_works_1() {
    let a = U256::from_hex_unchecked("1");
    let b = U256::from_hex_unchecked("10");
    assert_eq!(b >> 4, a);
}

#[test]
fn shift_right_on_256_bit_integer_works_2() {
    let a = U256::from_hex_unchecked("10");
    let b = U256::from_hex_unchecked("1000");
    assert_eq!(&b >> 8, a);
}

#[test]
fn shift_right_on_256_bit_integer_works_3() {
    let a = U256::from_hex_unchecked("e45542992b6844553f3cb1c5ac33e7fa5");
    let b = U256::from_hex_unchecked("391550a64ada11154fcf2c716b0cf9fe940");
    assert_eq!(b >> 6, a);
}

#[test]
fn shift_right_on_256_bit_integer_works_4() {
    let a = U256::from_hex_unchecked("390aa99bead76bc0093b1bc1a8101f5ce");
    let b = U256::from_hex_unchecked(
        "72155337d5aed7801276378350203eb9c0000000000000000000000000000000",
    );
    assert_eq!(&b >> 125, a);
}

#[test]
fn shift_right_on_256_bit_integer_works_5() {
    let a = U256::from_hex_unchecked(
        "ba6ab46f9a9a2f20e4061b67ce4d8c3da98091cf990d7b14ef47ffe27370abbd",
    );
    let b = U256::from_hex_unchecked("174d568df35345e41c80c36cf9c");
    assert_eq!(a >> 151, b);
}

#[test]
fn shift_right_on_256_bit_integer_works_6() {
    let a = U256::from_hex_unchecked(
        "076c075d2f65e39b9ecdde8bf6f8c94241962ce0f557b7739673200c777152eb",
    );
    let b = U256::from_hex_unchecked("ed80eba5ecbc7373d9bbd17edf19284832c59c");
    assert_eq!(&a >> 99, b);
}

#[test]
fn shift_right_on_256_bit_integer_works_7() {
    let a = U256::from_hex_unchecked(
        "6a9ce35d8940a5ebd29604ce9a182ade76f03f7e9965760b84a8cfd1d3dd2e61",
    );
    let b = U256::from_hex_unchecked("6a9ce35d8940a5eb");
    assert_eq!(a >> (64 * 3), b);
}

#[test]
fn shift_right_on_256_bit_integer_works_8() {
    let a = U256::from_hex_unchecked(
        "5322c128ec84081b6c376c108ebd7fd36bbd44f71ee5e6ad6bcb3dd1c5265bd7",
    );
    let b = U256::from_hex_unchecked("5322c128ec84081b6c376c108ebd7fd3");
    assert_eq!(a >> (64 * 2), b);
}

#[test]
#[cfg(feature = "alloc")]
fn to_be_bytes_works() {
    let number = U256::from_u64(1);
    let expected_bytes = [
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 1,
    ];

    assert_eq!(number.to_bytes_be(), expected_bytes);
}

#[test]
#[cfg(feature = "alloc")]
fn to_le_bytes_works() {
    let number = U256::from_u64(1);
    let expected_bytes = [
        1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0,
    ];

    assert_eq!(number.to_bytes_le(), expected_bytes);
}

#[test]
fn from_bytes_be_works() {
    let bytes = [
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 1,
    ];
    let expected_number = U256::from_u64(1);
    assert_eq!(U256::from_bytes_be(&bytes).unwrap(), expected_number);
}

#[test]
fn from_bytes_le_works() {
    let bytes = [
        1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0,
    ];
    let expected_number = U256::from_u64(1);
    assert_eq!(U256::from_bytes_le(&bytes).unwrap(), expected_number);
}

#[test]
fn shr_inplace_works_1() {
    let mut n = UnsignedInteger::<3>::from(4u64);
    n >>= 1;

    assert_eq!(n, UnsignedInteger::<3>::from(2u64));
}

#[test]
fn shr_inplace_on_256_bit_integer_works_1() {
    let a = U256::from_hex_unchecked("e45542992b6844553f3cb1c5ac33e7fa5");
    let mut b = U256::from_hex_unchecked("391550a64ada11154fcf2c716b0cf9fe940");
    b >>= 6;
    assert_eq!(a, b);
}

#[test]
fn shr_inplace_on_254_bit_integer_works_2() {
    let a = U256::from_hex_unchecked("390aa99bead76bc0093b1bc1a8101f5ce");
    let mut b = U256::from_hex_unchecked(
        "72155337d5aed7801276378350203eb9c0000000000000000000000000000000",
    );
    b >>= 125;
    assert_eq!(a, b);
}

#[test]
fn shr_inplace_on_256_bit_integer_works_3() {
    let a = U256::from_hex_unchecked("2ed786ab132f0b5b0cacd385dd51de3a");
    let mut b = U256::from_hex_unchecked(
        "2ed786ab132f0b5b0cacd385dd51de3a00000000000000000000000000000000",
    );
    b >>= 64 * 2;
    assert_eq!(a, b);
}

#[test]
fn shr_inplace_on_256_bit_integer_works_4() {
    let a = U256::from_hex_unchecked("90823e0bd707f");
    let mut b =
        U256::from_hex_unchecked("90823e0bd707f000000000000000000000000000000000000000000000000");
    b >>= 64 * 3;
    assert_eq!(a, b);
}

#[test]
fn shr_inplace_on_256_bit_integer_works_5() {
    let a = U256::from_hex_unchecked("24208f");
    let mut b =
        U256::from_hex_unchecked("90823e0bd707f000000000000000000000000000000000000000000000000");
    b >>= 222;
    assert_eq!(a, b);
}

#[test]
fn multiplying_and_dividing_for_number_is_number_with_remainder_0() {
    let a = U256::from_u128(12678920202929299999999999282828);
    let b = U256::from_u128(9000000000000);
    assert_eq!((a * b).div_rem(&b), (a, U256::from_u64(0)));
}

#[test]
fn unsigned_int_8_div_rem_3_is_2_2() {
    let a: UnsignedInteger<4> = U256::from_u64(8);
    let b = U256::from_u64(3);
    assert_eq!(a.div_rem(&b), (U256::from_u64(2), U256::from_u64(2)));
}

#[test]
fn unsigned_int_500721_div_rem_5_is_100144_1() {
    let a = U256::from_u64(500721);
    let b = U256::from_u64(5);
    assert_eq!(a.div_rem(&b), (U256::from_u64(100144), U256::from_u64(1)));
}

#[test]
fn div_rem_works_with_big_numbers() {
    let a = U256::from_u128(4758402376589578934275873583589345);
    let b = U256::from_u128(43950384634609);
    assert_eq!(
        a.div_rem(&b),
        (
            U256::from_u128(108267593472721187331),
            U256::from_u128(12368508650766)
        )
    );
}

#[cfg(feature = "std")]
#[test]
fn to_hex_test() {
    let a = U256::from_hex_unchecked("390aa99bead76bc0093b1bc1a8101f5ce");
    assert_eq!(U256::to_hex(&a), "390AA99BEAD76BC0093B1BC1A8101F5CE")
}
