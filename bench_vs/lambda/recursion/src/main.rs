//! Naive recursion guest: verifies an inner lambda-vm proof inside the VM.
//!
//! Private input layout: a 16-byte `"LVMR" + version + reserved` prefix
//! followed by an rkyv archive of `lambda_vm_prover::RecursionInput`
//! `{ vm_proof, inner_elf, options, vkey }`. The prefix tags the format so the
//! guest rejects a wrong-format blob before the unsafe access; sized to a
//! multiple of 16, it keeps the archive 16-aligned at the executor's aligned
//! payload base (`PRIVATE_INPUT_START + 16`). The proof is verified **in
//! place** via `verify_recursion_blob` — no deserialization pass, no owned
//! `VmProof`. Commits `vk_digest ‖ inner public output` on success: every
//! input here is prover-supplied, so soundness comes from the outer verifier
//! checking the committed digest against one derived from the trusted inner
//! ELF.
//!
//! Not `no_std` (std/alloc are available — `build-std` provides them, and the
//! prover links as a normal std crate; its prove-side code is dead-code
//! eliminated since we only call `verify`). Like every other allocating guest
//! it is `#![no_main]` and uses the syscalls crate's global allocator (a large
//! `TlsfHeap`), initialized first thing in `main` — `verify` allocates far more
//! than the target's default heap provides.

#![no_main]

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

    // Zero-copy: borrow the blob straight from the mapped private-input region.
    // The payload base and prefix are both 16-aligned, so the archive sits at a
    // 16-aligned guest address and the verifier's in-place loads don't trap.
    let blob = lambda_vm_syscalls::syscalls::get_private_input_slice();
    lambda_vm_prover::profile_markers::step_marker::<
        { lambda_vm_prover::profile_markers::STEP_DECODE_DONE },
    >();

    let verification = lambda_vm_prover::verify_recursion_blob(blob).expect("verify errored");
    assert!(verification.ok, "inner proof failed verification");

    let mut output = Vec::with_capacity(32 + verification.public_output.len());
    output.extend_from_slice(&verification.vk_digest);
    output.extend_from_slice(verification.public_output);
    lambda_vm_syscalls::syscalls::commit(&output);
    lambda_vm_syscalls::syscalls::sys_halt();
}
