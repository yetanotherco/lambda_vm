use crate::errors::DeserializationError;

use crate::errors::ByteConversionError;
/// A trait for converting an element to and from its byte representation and
/// for getting an element from its byte representation in big-endian or
/// little-endian order.
pub trait ByteConversion {
    /// Returns the byte representation of the element in big-endian order.}
    #[cfg(feature = "alloc")]
    fn to_bytes_be(&self) -> alloc::vec::Vec<u8>;

    /// Returns the byte representation of the element in little-endian order.
    #[cfg(feature = "alloc")]
    fn to_bytes_le(&self) -> alloc::vec::Vec<u8>;

    /// Returns the element from its byte representation in big-endian order.
    fn from_bytes_be(bytes: &[u8]) -> Result<Self, ByteConversionError>
    where
        Self: Sized;

    /// Returns the element from its byte representation in little-endian order.
    fn from_bytes_le(bytes: &[u8]) -> Result<Self, ByteConversionError>
    where
        Self: Sized;
}

/// Serialize function without args
/// Used for serialization when formatting options are not relevant
#[cfg(feature = "alloc")]
pub trait AsBytes {
    /// Default serialize without args
    fn as_bytes(&self) -> alloc::vec::Vec<u8>;
}

/// Zero-allocation byte serialization for Merkle tree hashing.
///
/// Unlike `AsBytes::as_bytes()` which returns `Vec<u8>` (heap allocation per call),
/// this trait writes bytes directly into a caller-provided buffer. For Merkle trees
/// with millions of field elements, this eliminates millions of 8-byte allocations.
pub trait WriteBytes {
    /// Byte length of this element's big-endian representation.
    const BYTE_LEN: usize;
    /// Write big-endian bytes into `buf[..BYTE_LEN]`.
    fn write_bytes_be(&self, buf: &mut [u8]);
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
    #[cfg(feature = "alloc")]
    fn to_bytes_be(&self) -> alloc::vec::Vec<u8> {
        self.to_be_bytes().to_vec()
    }

    #[cfg(feature = "alloc")]
    fn to_bytes_le(&self) -> alloc::vec::Vec<u8> {
        self.to_le_bytes().to_vec()
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
