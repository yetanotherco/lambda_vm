//! Proof bundle for serialization and verification.
//!
//! This module provides [`ProofBundle`], a wrapper around [`MultiProof`] that includes
//! all necessary metadata for standalone verification. The bundle is serialized using
//! CBOR format for compact binary representation.
//!
//! # Format
//!
//! The proof bundle contains three main components:
//!
//! 1. **multi_proof** - The STARK multi-proof containing proofs for all VM tables:
//!    - CPU table: Main execution trace
//!    - Bitwise table: AND, OR, XOR operations
//!    - LT table: Less-than comparisons
//!    - MEMW table: Memory write operations
//!    - LOAD table: Memory load operations
//!
//! 2. **proof_options** - Parameters used for proof generation, including:
//!    - Security level (blowup factor, number of queries)
//!    - Coset offset for LDE domain
//!    - FRI folding factor
//!
//! 3. **metadata** - Information about the proven execution:
//!    - Format version (for future compatibility)
//!    - SHA3-256 hash of the ELF file (for integrity verification)
//!    - Number of execution steps
//!
//! # Versioning
//!
//! The `version` field in metadata allows for future format changes while
//! maintaining backward compatibility. Current version is 1.
//!
//! # Example
//!
//! ```ignore
//! use cli::proof_bundle::ProofBundle;
//!
//! // Create a bundle after proving
//! let bundle = ProofBundle::new(multi_proof, proof_options, elf_hash, num_steps);
//!
//! // Serialize to CBOR
//! let file = File::create("proof.cbor")?;
//! ciborium::into_writer(&bundle, BufWriter::new(file))?;
//!
//! // Deserialize from CBOR
//! let file = File::open("proof.cbor")?;
//! let bundle: ProofBundle = ciborium::from_reader(BufReader::new(file))?;
//! ```

use prover::tables::types::{GoldilocksExtension, GoldilocksField};
use serde::{Deserialize, Serialize};
use stark::proof::options::ProofOptions;
use stark::proof::stark::MultiProof;

/// Current version of the proof bundle format.
pub const PROOF_BUNDLE_VERSION: u32 = 1;

/// Metadata about the proof.
///
/// Contains information needed to identify and validate the proof
/// without deserializing the full multi-proof structure.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProofMetadata {
    /// Version of the proof bundle format.
    ///
    /// Used for backward compatibility when the format evolves.
    /// Current version is 1.
    pub version: u32,

    /// SHA3-256 hash of the original ELF file.
    ///
    /// Can be used to verify that a proof corresponds to a specific program.
    /// The verifier can compute this hash independently to ensure the proof
    /// was generated for the expected program.
    pub elf_hash: [u8; 32],

    /// Number of RISC-V instructions executed.
    ///
    /// This is the trace length before padding to a power of two.
    /// Useful for understanding proof complexity and debugging.
    pub num_steps: usize,
}

/// A proof bundle containing the multi-proof, options, and metadata.
///
/// This is the complete serializable format for Lambda VM proofs.
/// It contains everything needed to verify execution of a RISC-V program
/// without access to the original program or execution trace.
///
/// The bundle is designed to be:
/// - **Self-contained**: All verification parameters are included
/// - **Compact**: Uses CBOR binary serialization
/// - **Versioned**: Includes format version for future compatibility
///
/// # Verification
///
/// To verify a proof bundle:
/// 1. Deserialize the bundle from CBOR
/// 2. Reconstruct AIRs using the embedded `proof_options`
/// 3. Call `Verifier::multi_verify` with the AIRs and `multi_proof`
///
/// The verifier must use the same bitwise table preprocessed commitment
/// that was used during proof generation.
#[derive(Debug, Serialize, Deserialize)]
#[serde(bound = "")]
pub struct ProofBundle {
    /// The multi-proof containing STARK proofs for all VM tables.
    ///
    /// This includes proofs for:
    /// - CPU table (main execution)
    /// - Bitwise table (AND, OR, XOR)
    /// - LT table (less-than comparisons)
    /// - MEMW table (memory writes)
    /// - LOAD table (memory loads)
    pub multi_proof: MultiProof<GoldilocksField, GoldilocksExtension, ()>,

    /// Proof options used during proof generation.
    ///
    /// These must be used to reconstruct the AIRs during verification.
    /// Includes security parameters like blowup factor and query count.
    pub proof_options: ProofOptions,

    /// Metadata about the proven execution.
    pub metadata: ProofMetadata,
}

impl ProofBundle {
    /// Creates a new proof bundle.
    ///
    /// # Arguments
    ///
    /// * `multi_proof` - The multi-proof from `Prover::multi_prove`
    /// * `proof_options` - The options used for proof generation
    /// * `elf_hash` - SHA3-256 hash of the ELF file
    /// * `num_steps` - Number of instructions executed
    ///
    /// # Example
    ///
    /// ```ignore
    /// let bundle = ProofBundle::new(
    ///     multi_proof,
    ///     proof_options,
    ///     elf_hash,
    ///     result.logs.len(),
    /// );
    /// ```
    pub fn new(
        multi_proof: MultiProof<GoldilocksField, GoldilocksExtension, ()>,
        proof_options: ProofOptions,
        elf_hash: [u8; 32],
        num_steps: usize,
    ) -> Self {
        Self {
            multi_proof,
            proof_options,
            metadata: ProofMetadata {
                version: PROOF_BUNDLE_VERSION,
                elf_hash,
                num_steps,
            },
        }
    }

    /// Returns the format version of this bundle.
    #[inline]
    #[allow(dead_code)]
    pub fn version(&self) -> u32 {
        self.metadata.version
    }

    /// Returns the SHA3-256 hash of the original ELF file.
    #[inline]
    #[allow(dead_code)]
    pub fn elf_hash(&self) -> &[u8; 32] {
        &self.metadata.elf_hash
    }

    /// Returns the number of instructions that were executed.
    #[inline]
    #[allow(dead_code)]
    pub fn num_steps(&self) -> usize {
        self.metadata.num_steps
    }
}
