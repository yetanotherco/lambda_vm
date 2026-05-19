//! SP1 guest that runs lambda-vm's `verify_with_options` on a single proof.
//!
//! Input layout (postcard-encoded `Vec<u8>` written via `SP1Stdin::write_vec`):
//!   `(VmProof, Vec<u8>, ProofOptions)`
//! where the inner `Vec<u8>` is the inner program's ELF bytes.
//!
//! Output: commits `[1u8]` on successful verify; the guest panics otherwise.
//!
//! Caveats:
//! - The verifier hashes through the `keccak` crate. SP1 has a Keccak
//!   precompile but it patches `tiny-keccak`, not `keccak`. We don't patch
//!   here, so Keccak runs as software inside the guest. Cycle counts will be
//!   inflated by that overhead. Worth keeping in mind when interpreting the
//!   number relative to lambda-vm's in-VM count.

#![no_main]

extern crate alloc;

use alloc::vec::Vec;

use lambda_vm_prover::{ProofOptions, VmProof};

sp1_zkvm::entrypoint!(main);

pub fn main() {
    let blob = sp1_zkvm::io::read_vec();
    let (vm_proof, inner_elf, options): (VmProof, Vec<u8>, ProofOptions) =
        postcard::from_bytes(&blob).expect("failed to deserialize input");
    let ok = lambda_vm_prover::verify_with_options(&vm_proof, &inner_elf, &options)
        .expect("verify errored");
    assert!(ok, "inner proof failed verification");
    sp1_zkvm::io::commit_slice(&[1u8]);
}
