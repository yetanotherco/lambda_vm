/// Implementation of FieldElement, a generic element of a field.
pub mod element;
pub mod errors;
/// Quadratic and cubic extensions of the Goldilocks field.
pub mod extensions_goldilocks;
/// Optimized Goldilocks field p = 2^64 - 2^32 + 1 (no Montgomery form)
pub mod goldilocks;
/// Field for test purposes.
pub mod test_fields;
/// Common behaviour for field elements.
pub mod traits;
