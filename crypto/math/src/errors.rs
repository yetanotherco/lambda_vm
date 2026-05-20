#[derive(Debug, PartialEq, Eq)]
pub enum ByteConversionError {
    FromBEBytesError,
    FromLEBytesError,
    ValueNotReduced,
}

#[derive(Debug, PartialEq, Eq)]
pub enum CreationError {
    InvalidHexString,
    HexStringIsTooBig,
    CanonicalOutOfRange,
    EmptyString,
}
