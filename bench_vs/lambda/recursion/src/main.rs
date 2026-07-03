//! Naive recursion guest: verifies an inner lambda-vm proof inside the VM.
//!
//! Private input layout (postcard-encoded):
//!   `(VmProof, Vec<u8>, ProofOptions, VmVerifyingKey)`
//! where the `Vec<u8>` holds the inner program's ELF bytes and `ProofOptions`
//! specifies the parameters the inner prover used. On success commits a
//! postcard-encoded [`RecursionCommitment`]: every input here is
//! prover-supplied, so soundness comes from an outer verifier passing it to
//! `verify_recursion_commitment` with the trusted inner ELF.
//!
//! Not `no_std` (std/alloc are available — `build-std` provides them, and the
//! prover links as a normal std crate; its prove-side code is dead-code
//! eliminated since we only call `verify`). Like every other allocating guest
//! it is `#![no_main]` and uses the syscalls crate's global allocator (a large
//! `TlsfHeap`), initialized first thing in `main` — `verify` allocates far more
//! than the target's default heap provides.

#![no_main]

use lambda_vm_prover::{ProofOptions, RecursionCommitment, VmProof, VmVerifyingKey};

#[unsafe(export_name = "main")]
pub fn main() -> ! {
    lambda_vm_syscalls::allocator::init_allocator();

    // Install panic handler to make sure any OOM is because verifying itself is
    // expensive rather than panics causing stack unwinding, which itself is very
    // expensive in the guest.
    const PANIC_MSG: &str = "PANICKED";
    std::panic::set_hook(Box::new(|_| unsafe {
        lambda_vm_syscalls::syscalls::sys_panic(PANIC_MSG.as_ptr(), PANIC_MSG.len())
    }));

    let blob = lambda_vm_syscalls::syscalls::get_private_input();
    let (vm_proof, inner_elf, options, vkey): (VmProof, Vec<u8>, ProofOptions, VmVerifyingKey) =
        postcard::from_bytes(&blob).expect("failed to deserialize recursion input");
    lambda_vm_prover::profile_markers::step_marker::<
        { lambda_vm_prover::profile_markers::STEP_DECODE_DONE },
    >();

    let ok = lambda_vm_prover::verify_with_options_with_vkey(
        &vm_proof,
        &inner_elf,
        &options,
        None,
        None,
        Some(&vkey),
    )
    .expect("verify errored");
    assert!(ok, "inner proof failed verification");

    // `vm_proof.vk_digest` was just checked equal to `vkey.compute_digest()`
    // inside verify, so reuse it instead of re-serializing and re-hashing.
    let commitment = RecursionCommitment {
        elf_digest: lambda_vm_prover::elf_digest(&inner_elf),
        vk_digest: vm_proof.vk_digest,
        options,
        table_counts: vm_proof.table_counts,
        num_private_input_pages: vm_proof.num_private_input_pages,
        runtime_page_ranges: vm_proof.runtime_page_ranges,
        public_output: vm_proof.public_output,
    };
    let output = postcard::to_allocvec(&commitment).expect("failed to serialize commitment");
    lambda_vm_syscalls::syscalls::commit(&output);
    lambda_vm_syscalls::syscalls::sys_halt();
}
