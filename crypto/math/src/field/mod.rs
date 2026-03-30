/// Implementation of FieldElement, a generic element of a field.
pub mod element;
pub mod errors;
/// Implementation of quadratic extensions of fields.
pub mod extensions;
/// Implementation of particular cases of fields.
pub mod fields;
/// PackedField trait for SIMD-vectorized field arithmetic.
pub mod packed;
/// Field for test purposes.
pub mod test_fields;
/// Common behaviour for field elements.
pub mod traits;
