//! Deserialize-only counterpart to the recursion guest.
//!
//! Reads the same private-input blob as `recursion-bench`, postcard-decodes
//! `(VmProof, Vec<u8>, ProofOptions)`, then commits and halts — without ever
//! calling `verify_with_options`. The cycle delta between this guest and
//! `recursion-bench` is the actual cost of the STARK verifier inside the VM.
//!
//! Mirrors the recursion guest's std setup (build-std + `lambda_vm_syscalls`)
//! so the two differ only in the verify call.

#![no_main]

use lambda_vm_prover::{ProofOptions, VmProof};

#[unsafe(export_name = "main")]
pub fn main() -> ! {
    lambda_vm_syscalls::allocator::init_allocator();

    const PANIC_MSG: &str = "PANICKED";
    std::panic::set_hook(Box::new(|_| unsafe {
        lambda_vm_syscalls::syscalls::sys_panic(PANIC_MSG.as_ptr(), PANIC_MSG.len())
    }));

    let blob = lambda_vm_syscalls::syscalls::get_private_input();
    let decoded: (VmProof, Vec<u8>, ProofOptions) =
        postcard::from_bytes(&blob).expect("failed to deserialize recursion input");

    // Tie the committed byte to the decoded value so LLVM can't elide the decode.
    let marker = decoded.2.blowup_factor ^ *decoded.1.first().unwrap_or(&0);
    lambda_vm_syscalls::syscalls::commit(&[marker]);
    lambda_vm_syscalls::syscalls::sys_halt();
}
