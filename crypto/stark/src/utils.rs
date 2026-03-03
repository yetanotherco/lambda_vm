use crypto::merkle_tree::proof::Proof;
use math::errors::DeserializationError;

use super::config::Commitment;

#[cfg(feature = "quaternary-merkle")]
const SIBLINGS_PER_LEVEL: usize = 3;
#[cfg(not(feature = "quaternary-merkle"))]
const SIBLINGS_PER_LEVEL: usize = 1;

pub fn serialize_proof(proof: &Proof<Commitment>) -> Vec<u8> {
    let mut bytes = vec![];
    bytes.extend(proof.merkle_path.len().to_be_bytes());
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

    for _ in 0..merkle_path_len {
        let mut siblings = Vec::with_capacity(SIBLINGS_PER_LEVEL);
        for _ in 0..SIBLINGS_PER_LEVEL {
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
