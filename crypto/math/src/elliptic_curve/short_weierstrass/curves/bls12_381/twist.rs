use crate::cyclic_group::IsGroup;
use crate::elliptic_curve::short_weierstrass::point::ShortWeierstrassProjectivePoint;
use crate::elliptic_curve::traits::IsEllipticCurve;
use crate::unsigned_integer::element::U384;
use crate::{
    elliptic_curve::short_weierstrass::traits::IsShortWeierstrass, field::element::FieldElement,
};

use super::field_extension::{Degree2ExtensionField, Degree12ExtensionField};

const GENERATOR_X_0: U384 = U384::from_hex_unchecked(
    "024aa2b2f08f0a91260805272dc51051c6e47ad4fa403b02b4510b647ae3d1770bac0326a805bbefd48056c8c121bdb8",
);
const GENERATOR_X_1: U384 = U384::from_hex_unchecked(
    "13e02b6052719f607dacd3a088274f65596bd0d09920b61ab5da61bbdc7f5049334cf11213945d57e5ac7d055d042b7e",
);
const GENERATOR_Y_0: U384 = U384::from_hex_unchecked(
    "0ce5d527727d6e118cc9cdc6da2e351aadfd9baa8cbdd3a76d429a695160d12c923ac9cc3baca289e193548608b82801",
);
const GENERATOR_Y_1: U384 = U384::from_hex_unchecked(
    "0606c4a02ea734cc32acd2b02bc28b99cb3e287e85a763af267492ab572e99ab3f370d275cec1da1aaa9075ff05f79be",
);

/// The description of the curve.
#[derive(Clone, Debug)]
pub struct BLS12381TwistCurve;

impl IsEllipticCurve for BLS12381TwistCurve {
    type BaseField = Degree2ExtensionField;
    type PointRepresentation = ShortWeierstrassProjectivePoint<Self>;

    fn generator() -> Self::PointRepresentation {
        // SAFETY:
        // - The generator point is mathematically verified to be a valid point on the curve.
        // - `unwrap()` is safe because the provided coordinates satisfy the curve equation.
        let point = Self::PointRepresentation::new([
            FieldElement::new([
                FieldElement::new(GENERATOR_X_0),
                FieldElement::new(GENERATOR_X_1),
            ]),
            FieldElement::new([
                FieldElement::new(GENERATOR_Y_0),
                FieldElement::new(GENERATOR_Y_1),
            ]),
            FieldElement::one(),
        ]);
        point.unwrap()
    }
}

impl IsShortWeierstrass for BLS12381TwistCurve {
    fn a() -> FieldElement<Self::BaseField> {
        FieldElement::zero()
    }

    fn b() -> FieldElement<Self::BaseField> {
        FieldElement::new([FieldElement::from(4), FieldElement::from(4)])
    }
}

impl ShortWeierstrassProjectivePoint<BLS12381TwistCurve> {
    /// This function is related to the map ψ: E_twist(𝔽p²) -> E(𝔽p¹²).
    /// Given an affine point G in E_twist(𝔽p²) returns x, y such that
    /// ψ(G) = (x', y', 1) with x' = x * x'' and y' = y * y''
    /// for some x'', y'' in 𝔽p².
    /// This is meant only to be used in the miller loop of the
    /// ate pairing before the final exponentiation.
    /// This is because elements in 𝔽p² raised to that
    /// power are 1 and so the final result of the ate pairing
    /// doens't depend on having this function output the exact
    /// values of x' and y'. And it is enough to work with x and y.
    pub fn to_fp12_unnormalized(&self) -> [FieldElement<Degree12ExtensionField>; 2] {
        if self.is_neutral_element() {
            [FieldElement::zero(), FieldElement::one()]
        } else {
            let [qx, qy, _] = self.coordinates();

            let result_x = FieldElement::new([
                FieldElement::new([FieldElement::zero(), qx.clone(), FieldElement::zero()]),
                FieldElement::zero(),
            ]);

            let result_y = FieldElement::new([
                FieldElement::zero(),
                FieldElement::new([FieldElement::zero(), qy.clone(), FieldElement::zero()]),
            ]);

            [result_x, result_y]
        }
    }
}
