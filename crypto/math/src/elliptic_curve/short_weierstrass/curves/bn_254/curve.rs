use super::{
    field_extension::{BN254PrimeField, Degree2ExtensionField},
    pairing::{GAMMA_12, GAMMA_13, X},
    twist::BN254TwistCurve,
};
use crate::cyclic_group::IsGroup;
use crate::elliptic_curve::short_weierstrass::point::ShortWeierstrassProjectivePoint;
use crate::elliptic_curve::traits::IsEllipticCurve;
use crate::{
    elliptic_curve::short_weierstrass::traits::IsShortWeierstrass, field::element::FieldElement,
};

pub type BN254FieldElement = FieldElement<BN254PrimeField>;
pub type BN254TwistCurveFieldElement = FieldElement<Degree2ExtensionField>;

#[derive(Clone, Debug)]
pub struct BN254Curve;

impl IsEllipticCurve for BN254Curve {
    type BaseField = BN254PrimeField;
    type PointRepresentation = ShortWeierstrassProjectivePoint<Self>;

    /// Returns the generator point of the BN254 curve.
    ///
    /// # Safety
    ///
    /// - The generator point is mathematically verified to be a valid point on the curve.
    /// - `unwrap()` is safe because the provided coordinates satisfy the curve equation.
    fn generator() -> Self::PointRepresentation {
        // SAFETY:
        // - The generator coordinates `(1, 2, 1)` are **predefined** and belong to the BN254 curve.
        // - `unwrap()` is safe because we **ensure** the input values satisfy the curve equation.
        let point = Self::PointRepresentation::new([
            FieldElement::<Self::BaseField>::one(),
            FieldElement::<Self::BaseField>::from(2),
            FieldElement::one(),
        ]);
        point.unwrap()
    }
}

impl IsShortWeierstrass for BN254Curve {
    fn a() -> FieldElement<Self::BaseField> {
        FieldElement::from(0)
    }

    fn b() -> FieldElement<Self::BaseField> {
        FieldElement::from(3)
    }
}

impl ShortWeierstrassProjectivePoint<BN254Curve> {
    pub fn is_in_subgroup(&self) -> bool {
        true
    }
}

impl ShortWeierstrassProjectivePoint<BN254TwistCurve> {
    /// phi morphism used to G2 subgroup check for twisted curve.
    /// We also use phi at the last lines of the Miller Loop of the pairing.
    /// phi(q) = (x^p, y^p, z^p), where (x, y, z) are the projective coordinates of q.
    /// See https://hackmd.io/@Wimet/ry7z1Xj-2#Subgroup-Checks.
    ///
    /// # Safety
    ///
    /// - The function assumes `self` is a valid point on the BN254 twist curve.
    /// - The transformation follows a known isomorphism and preserves validity.
    pub fn phi(&self) -> Self {
        let [x, y, z] = self.coordinates();
        // SAFETY:
        // - `conjugate()` preserves the validity of the field element.
        // - `unwrap()` is safe because the transformation follows
        //   **a known valid isomorphism** between the twist and E.
        let point = Self::new([
            x.conjugate() * GAMMA_12,
            y.conjugate() * GAMMA_13,
            z.conjugate(),
        ]);
        point.unwrap()
    }

    // Checks if a G2 point is in the subgroup of the twisted curve.
    pub fn is_in_subgroup(&self) -> bool {
        let q_times_x = &self.operate_with_self(X);
        let q_times_x_plus_1 = &self.operate_with(q_times_x);
        let q_times_2x = q_times_x.double();

        // (x+1)Q + phi(xQ) + phi(phi(xQ)) == phi(phi(phi(2xQ)))
        q_times_x_plus_1.operate_with(&q_times_x.phi().operate_with(&q_times_x.phi().phi()))
            == q_times_2x.phi().phi().phi()
    }
}
