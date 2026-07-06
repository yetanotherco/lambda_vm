//! Naive recursion guest: verifies an inner lambda-vm proof inside the VM.
//!
//! Private input layout (postcard-encoded):
//!   `(VmProof, Vec<u8>, ProofOptions)`
//! where the `Vec<u8>` holds the inner program's ELF bytes and `ProofOptions`
//! specifies the parameters the inner prover used. Commits `[1]` on success.
//!
//! Not `no_std` (std/alloc are available — `build-std` provides them, and the
//! prover links as a normal std crate; its prove-side code is dead-code
//! eliminated since we only call `verify`). Like every other allocating guest
//! it is `#![no_main]` and uses the syscalls crate's global allocator (a large
//! `TlsfHeap`), initialized first thing in `main` — `verify` allocates far more
//! than the target's default heap provides.

#![no_main]

use lambda_vm_prover::{ProofOptions, VmProof};

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
    let (vm_proof, inner_elf, options): (VmProof, Vec<u8>, ProofOptions) =
        postcard::from_bytes(&blob).expect("failed to deserialize recursion input");
    lambda_vm_prover::profile_markers::step_marker::<
        { lambda_vm_prover::profile_markers::STEP_DECODE_DONE },
    >();

    let ok = lambda_vm_prover::verify_with_options(&vm_proof, &inner_elf, &options, None, None)
        .expect("verify errored");
    assert!(ok, "inner proof failed verification");

    lambda_vm_syscalls::syscalls::commit(&[1u8]);
    lambda_vm_syscalls::syscalls::sys_halt();
}
