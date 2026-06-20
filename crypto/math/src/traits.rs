use crate::errors::{ByteConversionError, DeserializationError};
/// A trait for converting an element to and from its byte representation and
/// for getting an element from its byte representation in big-endian or
/// little-endian order.
pub trait ByteConversion {
    /// Byte length of the big-endian representation.
    const BYTE_LEN: usize;

    /// Fixed-length byte buffer returned by [`to_bytes_be`](Self::to_bytes_be)
    /// and [`to_bytes_le`](Self::to_bytes_le). For field elements this is a
    /// `[u8; BYTE_LEN]`, so serialization allocates nothing — a hot path in the
    /// Fiat-Shamir transcript and Merkle hashing. Borrow the bytes with
    /// `.as_ref()`; collect with `.as_ref().to_vec()` only when a `Vec` is
    /// actually required.
    type FixedBytes: AsRef<[u8]>;

    /// Returns the byte representation of the element in big-endian order.
    fn to_bytes_be(&self) -> Self::FixedBytes;

    /// Returns the byte representation of the element in little-endian order.
    fn to_bytes_le(&self) -> Self::FixedBytes;

    /// Returns the element from its byte representation in big-endian order.
    fn from_bytes_be(bytes: &[u8]) -> Result<Self, ByteConversionError>
    where
        Self: Sized;

    /// Returns the element from its byte representation in little-endian order.
    fn from_bytes_le(bytes: &[u8]) -> Result<Self, ByteConversionError>
    where
        Self: Sized;

    /// Write big-endian bytes into `buf[..BYTE_LEN]`.
    /// Override for zero-allocation performance in hot paths.
    fn write_bytes_be(&self, buf: &mut [u8]) {
        let bytes = self.to_bytes_be();
        let bytes = bytes.as_ref();
        buf[..bytes.len()].copy_from_slice(bytes);
    }
}

/// Serialize function without args
/// Used for serialization when formatting options are not relevant
#[cfg(feature = "alloc")]
pub trait AsBytes {
    /// Default serialize without args
    fn as_bytes(&self) -> alloc::vec::Vec<u8>;
}

#[cfg(feature = "alloc")]
impl AsBytes for u32 {
    fn as_bytes(&self) -> alloc::vec::Vec<u8> {
        self.to_le_bytes().to_vec()
    }
}

#[cfg(feature = "alloc")]
impl AsBytes for u64 {
    fn as_bytes(&self) -> alloc::vec::Vec<u8> {
        self.to_le_bytes().to_vec()
    }
}

impl ByteConversion for u64 {
    const BYTE_LEN: usize = 8;

    type FixedBytes = [u8; 8];

    fn to_bytes_be(&self) -> [u8; 8] {
        self.to_be_bytes()
    }

    fn to_bytes_le(&self) -> [u8; 8] {
        self.to_le_bytes()
    }

    fn from_bytes_be(bytes: &[u8]) -> Result<Self, ByteConversionError>
    where
        Self: Sized,
    {
        let needed_bytes = bytes
            .get(0..8)
            .ok_or(ByteConversionError::FromBEBytesError)?;
        Ok(u64::from_be_bytes(
            needed_bytes
                .try_into()
                .map_err(|_| ByteConversionError::FromBEBytesError)?,
        ))
    }

    fn from_bytes_le(bytes: &[u8]) -> Result<Self, ByteConversionError>
    where
        Self: Sized,
    {
        let needed_bytes = bytes
            .get(0..8)
            .ok_or(ByteConversionError::FromLEBytesError)?;
        Ok(u64::from_le_bytes(
            needed_bytes
                .try_into()
                .map_err(|_| ByteConversionError::FromLEBytesError)?,
        ))
    }
}

/// Deserialize function without args
pub trait Deserializable {
    fn deserialize(bytes: &[u8]) -> Result<Self, DeserializationError>
    where
        Self: Sized;
}
