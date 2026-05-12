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
//! All five preprocessed tables — BITWISE, DECODE, REGISTER, KECCAK_RC, and
//! every non-private-input PAGE — are cached here. `VmAirs::new_with_vkey`
//! prefers the vkey-supplied commitment over recomputing when a vkey is
//! provided. The `version` field exists so a vkey serialized against an
//! older layout produces a different `compute_digest()` and stops
//! validating.
//!
//! ## Security
//!
//! For this PR the verifying key is only a performance shortcut. The
//! verifier still relies on Fiat-Shamir: every preprocessed commitment the
//! prover used is bound into the proof's challenges, so a verifier that
//! consumes a tampered `vkey` field derives different challenges, the
//! openings stop matching, and verification fails. A future PR will
//! additionally embed `vkey.compute_digest()` in `VmProof` so vkey
//! substitution surfaces as an explicit error before any STARK work runs.

use alloc::vec::Vec;

use executor::elf::Elf;
use sha3::{Digest, Keccak256};
use stark::config::Commitment;
use stark::proof::options::ProofOptions;

use crate::tables::bitwise;
use crate::tables::decode;
use crate::tables::keccak_rc;
use crate::tables::page::{self, PageConfig};
use crate::tables::register;

/// Current `VmVerifyingKey` layout version. Bump whenever fields are added,
/// removed, or reordered so that vkeys serialized against an older layout
/// produce a different `compute_digest()` and stop validating.
pub const VKEY_VERSION: u32 = 3;

/// Placeholder commitment stored in [`VmVerifyingKey::pages`] for
/// private-input page slots, where there is no preprocessed commitment to
/// cache. The verifier never reads these slots (private-input pages have no
/// `with_preprocessed(...)` call in `VmAirs::new`).
const PRIVATE_INPUT_PAGE_PLACEHOLDER: Commitment = [0u8; 32];

/// Cached preprocessed-table commitments the verifier would otherwise
/// recompute on every call.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct VmVerifyingKey {
    /// Layout version. See [`VKEY_VERSION`].
    pub version: u32,
    /// Merkle root over the LDE of the bitwise preprocessed columns.
    /// Program-independent; depends only on `ProofOptions`.
    pub bitwise: Commitment,
    /// Merkle root over the LDE of the decode preprocessed columns.
    /// Program-dependent: derived from the inner ELF's instruction stream.
    pub decode: Commitment,
    /// Merkle root over the LDE of the register preprocessed columns.
    /// Program-dependent via the ELF's entry point.
    pub register: Commitment,
    /// Merkle root over the LDE of the keccak round-constants preprocessed
    /// columns. Program-independent; depends only on `ProofOptions`.
    pub keccak_rc: Commitment,
    /// Per-page preprocessed Merkle roots, indexed parallel to the
    /// `page_configs` slice the verifier reconstructs from the proof via
    /// [`crate::tables::trace_builder::Traces::page_configs_from_elf_and_runtime`].
    /// Private-input slots hold a zero placeholder and are never read by the
    /// verifier — they exist only to keep the index aligned with
    /// `page_configs`, which interleaves preprocessed and private-input pages.
    pub pages: Vec<Commitment>,
}

impl VmVerifyingKey {
    /// Derive the verifying key on the host.
    ///
    /// `elf` is read to derive the program-dependent commitments (DECODE
    /// from the instruction stream, REGISTER from `elf.entry_point`).
    ///
    /// `page_configs` must match exactly what the verifier will reconstruct
    /// from the proof — i.e. the output of
    /// `Traces::page_configs_from_elf_and_runtime(elf, runtime_page_ranges,
    /// num_private_input_pages)`. The host can call that helper with the
    /// values it already has after producing the inner proof.
    pub fn from_elf_and_options(
        elf: &Elf,
        options: &ProofOptions,
        page_configs: &[PageConfig],
    ) -> Self {
        let pages = page_configs
            .iter()
            .map(|config| {
                if config.is_private_input {
                    PRIVATE_INPUT_PAGE_PLACEHOLDER
                } else {
                    page::precomputed_commitment_cached(config, options)
                }
            })
            .collect();
        Self {
            version: VKEY_VERSION,
            bitwise: bitwise::preprocessed_commitment(options),
            decode: decode::commitment_from_elf(elf, options)
                .expect("decode commitment must compute"),
            register: register::preprocessed_commitment(options, elf.entry_point),
            keccak_rc: keccak_rc::preprocessed_commitment(options),
            pages,
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
