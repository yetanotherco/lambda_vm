//! Verifying key for the lambda-vm STARK verifier.
//!
//! Caches preprocessed-table Merkle commitments that the verifier would
//! otherwise recompute on every call. Mirrors the SP1 `MachineVerifyingKey`
//! pattern (preprocessed commitments derived once at setup, never recomputed
//! per-proof) and the prover-side companion in
//! <https://github.com/yetanotherco/lambda_vm/pull/282> (which caches the
//! same data on the prover side).
//!
//! ## Current scope
//!
//! Only the BITWISE preprocessed commitment is cached here. The other four
//! preprocessed tables (DECODE, KECCAK_RC, REGISTER, PAGE) are still
//! recomputed inside `VmAirs::new`; follow-up PRs will move them into this
//! struct one at a time. The `version` field exists so a vkey serialized
//! today does not accidentally validate against a future shape.
//!
//! ## Security
//!
//! For this PR the verifying key is only a performance shortcut. The
//! verifier still relies on Fiat-Shamir: the bitwise commitment the prover
//! used is bound into the proof's challenges, so a verifier that consumes a
//! tampered `vkey.bitwise` derives different challenges, the openings stop
//! matching, and verification fails. A future PR will additionally embed
//! `vkey.compute_digest()` in `VmProof` so vkey substitution surfaces as an
//! explicit error before any STARK work runs.

use executor::elf::Elf;
use sha3::{Digest, Keccak256};
use stark::config::Commitment;
use stark::proof::options::ProofOptions;

use crate::tables::bitwise;

/// Current `VmVerifyingKey` layout version. Bump whenever fields are added,
/// removed, or reordered so that vkeys serialized against an older layout
/// produce a different `compute_digest()` and stop validating.
pub const VKEY_VERSION: u32 = 1;

/// Cached preprocessed-table commitments the verifier would otherwise
/// recompute on every call.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct VmVerifyingKey {
    /// Layout version. See [`VKEY_VERSION`].
    pub version: u32,
    /// Merkle root over the LDE of the bitwise preprocessed columns.
    /// Program-independent; depends only on `ProofOptions`.
    pub bitwise: Commitment,
}

impl VmVerifyingKey {
    /// Derive the verifying key on the host.
    ///
    /// `elf` is unused for now (bitwise is program-independent) but stays in
    /// the signature so callers do not need to change when follow-up PRs
    /// fold in DECODE, REGISTER, and PAGE — which all depend on the ELF.
    pub fn from_elf_and_options(_elf: &Elf, options: &ProofOptions) -> Self {
        Self {
            version: VKEY_VERSION,
            bitwise: bitwise::preprocessed_commitment(options),
        }
    }

    /// Keccak256 fingerprint of the postcard-serialized vkey. Stable as long
    /// as the field layout (and [`VKEY_VERSION`]) does not change.
    pub fn compute_digest(&self) -> [u8; 32] {
        let bytes = postcard::to_allocvec(self)
            .expect("postcard serialization of VmVerifyingKey must succeed");
        let mut hasher = Keccak256::new();
        hasher.update(&bytes);
        hasher.finalize().into()
    }
}
