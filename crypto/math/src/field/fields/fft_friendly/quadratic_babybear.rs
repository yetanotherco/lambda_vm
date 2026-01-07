use crate::field::{
    element::FieldElement, extensions::quadratic::*,
    fields::fft_friendly::babybear::Babybear31PrimeField,
};

/// Quadratic field extension of Babybear
pub type QuadraticBabybearField =
    QuadraticExtensionField<Babybear31PrimeField, Babybear31PrimeField>;

impl HasQuadraticNonResidue<Babybear31PrimeField> for Babybear31PrimeField {
    fn residue() -> FieldElement<Babybear31PrimeField> {
        -FieldElement::one()
    }
}

/// Field element type for the quadratic extension of Babybear
pub type QuadraticBabybearFieldElement =
    QuadraticExtensionFieldElement<Babybear31PrimeField, Babybear31PrimeField>;
