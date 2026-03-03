use crypto::merkle_tree::proof::Proof;
use math::errors::DeserializationError;

use super::config::Commitment;

pub fn serialize_proof(proof: &Proof<Commitment>) -> Vec<u8> {
    let mut bytes = vec![];
    let num_levels = proof.merkle_path.len();
    bytes.extend(num_levels.to_be_bytes());
    // Encode siblings per level so deserialization is self-describing
    let siblings_per_level = if num_levels > 0 {
        proof.merkle_path[0].len()
    } else {
        0
    };
    bytes.extend(siblings_per_level.to_be_bytes());
    for siblings in &proof.merkle_path {
        for commitment in siblings {
            bytes.extend(commitment);
        }
    }
    bytes
}

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
    let siblings_per_level = usize::from_be_bytes(
        bytes
            .get(..8)
            .ok_or(DeserializationError::InvalidAmountOfBytes)?
            .try_into()
            .map_err(|_| DeserializationError::InvalidAmountOfBytes)?,
    );
    bytes = &bytes[8..];

    for _ in 0..merkle_path_len {
        let mut siblings = Vec::with_capacity(siblings_per_level);
        for _ in 0..siblings_per_level {
            let commitment: Commitment = bytes
                .get(..32)
                .ok_or(DeserializationError::InvalidAmountOfBytes)?
                .try_into()
                .map_err(|_| DeserializationError::InvalidAmountOfBytes)?;
            siblings.push(commitment);
            bytes = &bytes[32..];
        }
        merkle_path.push(siblings);
    }

    Ok((Proof { merkle_path }, bytes))
}
