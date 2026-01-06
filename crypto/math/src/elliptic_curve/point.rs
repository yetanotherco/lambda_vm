use crate::elliptic_curve::traits::IsEllipticCurve;
use crate::field::element::FieldElement;
use core::fmt::Debug;
/// Represents an elliptic curve point using the projective short Weierstrass form:
/// y^2 * z = x^3 + a * x * z^2 + b * z^3,
/// where `x`, `y` and `z` variables are field elements.
#[derive(Debug, Clone)]
pub struct ProjectivePoint<E: IsEllipticCurve> {
    pub value: [FieldElement<E::BaseField>; 3],
}

impl<E: IsEllipticCurve> ProjectivePoint<E> {
    /// Creates an elliptic curve point giving the projective [x: y: z] coordinates.
    pub const fn new(value: [FieldElement<E::BaseField>; 3]) -> Self {
        Self { value }
    }

    /// Returns the `x` coordinate of the point.
    pub fn x(&self) -> &FieldElement<E::BaseField> {
        &self.value[0]
    }

    /// Returns the `y` coordinate of the point.
    pub fn y(&self) -> &FieldElement<E::BaseField> {
        &self.value[1]
    }

    /// Returns the `z` coordinate of the point.
    pub fn z(&self) -> &FieldElement<E::BaseField> {
        &self.value[2]
    }

    /// Returns a tuple [x, y, z] with the coordinates of the point.
    pub fn coordinates(&self) -> &[FieldElement<E::BaseField>; 3] {
        &self.value
    }

    /// Creates the same point in affine coordinates. That is,
    /// returns [x / z: y / z: 1] where `self` is [x: y: z].
    /// Panics if `self` is the point at infinity.
    pub fn to_affine(&self) -> Self {
        let [x, y, z] = self.coordinates();
        // If it's the point at infinite
        if z == &FieldElement::zero() {
            // We make sure all the points at infinite have the same values
            return Self::new([
                FieldElement::zero(),
                FieldElement::one(),
                FieldElement::zero(),
            ]);
        };
        let inv_z = z.inv().unwrap();
        ProjectivePoint::new([x * &inv_z, y * inv_z, FieldElement::one()])
    }
}

impl<E: IsEllipticCurve> PartialEq for ProjectivePoint<E> {
    fn eq(&self, other: &Self) -> bool {
        let [px, py, pz] = self.coordinates();
        let [qx, qy, qz] = other.coordinates();
        (px * qz == pz * qx) && (py * qz == qy * pz)
    }
}

impl<E: IsEllipticCurve> Eq for ProjectivePoint<E> {}
#[derive(Debug, Clone)]

pub struct JacobianPoint<E: IsEllipticCurve> {
    pub value: [FieldElement<E::BaseField>; 3],
}

impl<E: IsEllipticCurve> JacobianPoint<E> {
    /// Creates an elliptic curve point giving the Jacobian [x: y: z] coordinates.
    pub const fn new(value: [FieldElement<E::BaseField>; 3]) -> Self {
        Self { value }
    }

    /// Returns the `x` coordinate of the point.
    pub fn x(&self) -> &FieldElement<E::BaseField> {
        &self.value[0]
    }

    /// Returns the `y` coordinate of the point.
    pub fn y(&self) -> &FieldElement<E::BaseField> {
        &self.value[1]
    }

    /// Returns the `z` coordinate of the point.
    pub fn z(&self) -> &FieldElement<E::BaseField> {
        &self.value[2]
    }

    /// Returns a tuple [x, y, z] with the coordinates of the point.
    pub fn coordinates(&self) -> &[FieldElement<E::BaseField>; 3] {
        &self.value
    }

    pub fn to_affine(&self) -> Self {
        let [x, y, z] = self.coordinates();
        // If it's the point at infinite
        if z == &FieldElement::zero() {
            // We make sure all the points at infinite have the same values
            return Self::new([
                FieldElement::one(),
                FieldElement::one(),
                FieldElement::zero(),
            ]);
        };
        let inv_z = z.inv().unwrap();
        let inv_z_square = inv_z.square();
        let inv_z_cube = &inv_z_square * &inv_z;
        JacobianPoint::new([x * inv_z_square, y * inv_z_cube, FieldElement::one()])
    }
}

impl<E: IsEllipticCurve> PartialEq for JacobianPoint<E> {
    fn eq(&self, other: &Self) -> bool {
        // In Jacobian coordinates, the equality of two points is defined as:
        // X1 * Z2^2 == X2 * Z1^2 y Y1 * Z2^3 == Y2 * Z1^3

        let [px, py, pz] = self.coordinates();
        let [qx, qy, qz] = other.coordinates();

        let zp_sq = pz.square();
        let zq_sq = qz.square();

        let zp_cu = &zp_sq * pz;
        let zq_cu = &zq_sq * qz;

        let xp_zq_sq = px * zq_sq;
        let xq_zp_sq = qx * zp_sq;

        let yp_zq_cu = py * zq_cu;
        let yq_zp_cu = qy * zp_cu;

        (xp_zq_sq == xq_zp_sq) && (yp_zq_cu == yq_zp_cu)
    }
}
