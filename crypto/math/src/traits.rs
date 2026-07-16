use crate::errors::{ByteConversionError, DeserializationError};
/// A trait for converting an element to and from its byte representation and
/// for getting an element from its byte representation in big-endian or
/// little-endian order.
pub trait ByteConversion {
    /// Byte length of the big-endian representation.
    const BYTE_LEN: usize;

    /// Returns the byte representation of the element in big-endian order.
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

    /// Write big-endian bytes into `buf[..BYTE_LEN]`.
    /// Override for zero-allocation performance in hot paths.
    #[cfg(feature = "alloc")]
    fn write_bytes_be(&self, buf: &mut [u8]) {
        let bytes = self.to_bytes_be();
        buf[..bytes.len()].copy_from_slice(&bytes);
    }
}

/// Serialize function without args
/// Used for serialization when formatting options are not relevant
#[cfg(feature = "alloc")]
pub trait AsBytes {
    /// Default serialize without args
    fn as_bytes(&self) -> alloc::vec::Vec<u8>;

    /// Streams the byte representation to `sink` without heap-allocating a `Vec`.
    /// Default falls back to `as_bytes`; override for zero-allocation hashing/transcript hot paths.
    ///
    /// An override must stream exactly the bytes `as_bytes` would return, in
    /// order; splitting them across several `sink` calls is fine, but the
    /// concatenation must be identical. Merkle leaf hashes and the Fiat-Shamir
    /// transcript take their input through here, so an override that disagrees
    /// with `as_bytes` silently changes commitments and challenges rather than
    /// failing to compile. `math/tests/stream_bytes_parity.rs` pins this for the
    /// Goldilocks fields.
    fn stream_bytes(&self, sink: &mut dyn FnMut(&[u8])) {
        sink(&self.as_bytes());
    }
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
