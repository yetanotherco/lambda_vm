use crypto::merkle_tree::proof::Proof;
use math::errors::DeserializationError;

use super::config::Commitment;

// ============================================================
// Keccak256 Implementation (default)
// ============================================================

#[cfg(not(feature = "poseidon2"))]
pub fn serialize_proof(proof: &Proof<Commitment>) -> Vec<u8> {
    let mut bytes = vec![];
    bytes.extend(proof.merkle_path.len().to_be_bytes());
    for commitment in &proof.merkle_path {
        bytes.extend(commitment);
    }
    bytes
}

#[cfg(not(feature = "poseidon2"))]
pub fn deserialize_proof(bytes: &[u8]) -> Result<(Proof<Commitment>, &[u8]), DeserializationError> {
    let mut bytes = bytes;
    let mut merkle_path = vec![];
    let merkle_path_len = usize::from_be_bytes(
        bytes
            .get(..8)
            .ok_or(DeserializationError::InvalidAmountOfBytes)?
            .try_into()
            .map_err(|_| DeserializationError::InvalidAmountOfBytes)?,
    );
    bytes = &bytes[8..];

    for _ in 0..merkle_path_len {
        let commitment = bytes
            .get(..32)
            .ok_or(DeserializationError::InvalidAmountOfBytes)?
            .try_into()
            .map_err(|_| DeserializationError::InvalidAmountOfBytes)?;
        merkle_path.push(commitment);
        bytes = &bytes[32..];
    }

    Ok((Proof { merkle_path }, bytes))
}

// ============================================================
// Poseidon2 Implementation (feature = "poseidon2")
// ============================================================
// Commitments are [Fp; 2] (128-bit) for security equivalent to Keccak256.

#[cfg(feature = "poseidon2")]
pub fn serialize_proof(proof: &Proof<Commitment>) -> Vec<u8> {
    use math::traits::AsBytes;

    let mut bytes = vec![];
    bytes.extend(proof.merkle_path.len().to_be_bytes());
    for commitment in &proof.merkle_path {
        // Each commitment is [Fp; 2], serialize both elements (16 bytes total)
        bytes.extend(commitment[0].as_bytes());
        bytes.extend(commitment[1].as_bytes());
    }
    bytes
}

#[cfg(feature = "poseidon2")]
pub fn deserialize_proof(bytes: &[u8]) -> Result<(Proof<Commitment>, &[u8]), DeserializationError> {
    use super::config::PoseidonFp;

    let mut bytes = bytes;
    let mut merkle_path = vec![];
    let merkle_path_len = usize::from_be_bytes(
        bytes
            .get(..8)
            .ok_or(DeserializationError::InvalidAmountOfBytes)?
            .try_into()
            .map_err(|_| DeserializationError::InvalidAmountOfBytes)?,
    );
    bytes = &bytes[8..];

    // Each Fp is 8 bytes (u64, big-endian), commitment is [Fp; 2] = 16 bytes
    const FP_SIZE: usize = 8;
    const COMMITMENT_SIZE: usize = FP_SIZE * 2; // 16 bytes for [Fp; 2]
    for _ in 0..merkle_path_len {
        let commitment_bytes = bytes
            .get(..COMMITMENT_SIZE)
            .ok_or(DeserializationError::InvalidAmountOfBytes)?;

        // Deserialize first Fp element
        let val0 = u64::from_be_bytes(
            commitment_bytes[..FP_SIZE]
                .try_into()
                .map_err(|_| DeserializationError::InvalidAmountOfBytes)?,
        );
        // Deserialize second Fp element
        let val1 = u64::from_be_bytes(
            commitment_bytes[FP_SIZE..COMMITMENT_SIZE]
                .try_into()
                .map_err(|_| DeserializationError::InvalidAmountOfBytes)?,
        );

        let commitment: Commitment = [PoseidonFp::from(val0), PoseidonFp::from(val1)];
        merkle_path.push(commitment);
        bytes = &bytes[COMMITMENT_SIZE..];
    }

    Ok((Proof { merkle_path }, bytes))
}
