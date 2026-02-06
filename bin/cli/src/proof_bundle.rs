//! Proof bundle for serialization and verification.

use prover::tables::types::{GoldilocksExtension, GoldilocksField};
use serde::{Deserialize, Serialize};
use stark::proof::options::ProofOptions;
use stark::proof::stark::MultiProof;

/// Current version of the proof bundle format.
pub const PROOF_BUNDLE_VERSION: u32 = 1;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProofMetadata {
    pub version: u32,
    pub elf_hash: [u8; 32],
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(bound = "")]
pub struct ProofBundle {
    pub multi_proof: MultiProof<GoldilocksField, GoldilocksExtension, ()>,
    pub proof_options: ProofOptions,
    pub metadata: ProofMetadata,
}

impl ProofBundle {
    pub fn new(
        multi_proof: MultiProof<GoldilocksField, GoldilocksExtension, ()>,
        proof_options: ProofOptions,
        elf_hash: [u8; 32],
    ) -> Self {
        Self {
            multi_proof,
            proof_options,
            metadata: ProofMetadata {
                version: PROOF_BUNDLE_VERSION,
                elf_hash,
            },
        }
    }
}
