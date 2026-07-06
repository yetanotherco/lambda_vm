//! Naive recursion guest: verifies an inner lambda-vm proof inside the VM.
//!
//! Private input layout (postcard-encoded):
//!   `(VmProof, Vec<u8>, Commitment, Vec<(u64, Commitment)>)`
//! where the `Vec<u8>` holds the inner program's ELF bytes, and the
//! `Commitment`/`Vec<(u64, Commitment)>` are the inner program's precomputed
//! DECODE and ELF-data-page commitments — supplied here instead of recomputed
//! in-VM (an ~45x cycle-count win: recomputing them via FFT+Merkle dominates
//! this guest's cost otherwise). They're untrusted, like every other private
//! input value: a wrong commitment diverges the inner proof's Fiat-Shamir
//! transcript, so `verify_with_options` returns `Ok(false)` rather than a
//! soundness gap.
//!
//! `ProofOptions` is deliberately NOT part of private input — it's fixed by
//! the `min`/`blowup8` Cargo feature this binary was built with (see
//! `recursion_proof_options` below). If it were attacker-supplied, a
//! malicious private input could pick trivially weak options (e.g. 1 FRI
//! query) and get the guest to accept + commit as if a real proof had been
//! checked, since the committed output can't otherwise convey what security
//! level was actually used.
//!
//! On success, commits `elf_digest(inner_elf) || decode_commitment ||
//! page_commitments` — the full identity of what was verified. Just the two
//! precomputed commitments wouldn't be enough: they only cover segment
//! *content* (executable segments / ELF-backed data pages), not e.g.
//! `entry_point`, so two ELFs could share both without being the same
//! program. `elf_digest` is the exact function `absorb_statement` already
//! binds into the transcript, reused as-is rather than inventing a second
//! identity scheme.
//!
//! Not `no_std` (std/alloc are available — `build-std` provides them, and the
//! prover links as a normal std crate; its prove-side code is dead-code
//! eliminated since we only call `verify`). Like every other allocating guest
//! it is `#![no_main]` and uses the syscalls crate's global allocator (a large
//! `TlsfHeap`), initialized first thing in `main` — `verify` allocates far more
//! than the target's default heap provides.

#![no_main]

#[cfg(feature = "blowup8")]
use lambda_vm_prover::GoldilocksCubicProofOptions;
use lambda_vm_prover::statement::elf_digest;
use lambda_vm_prover::{Commitment, ProofOptions, VmProof};

#[cfg(not(any(feature = "min", feature = "blowup8")))]
compile_error!("select exactly one of the `min`/`blowup8` features");
#[cfg(all(feature = "min", feature = "blowup8"))]
compile_error!("select exactly one of the `min`/`blowup8` features");

/// Smallest possible proof options (blowup=2, 1 query). Intentionally
/// insecure — for cheap diagnostics, not soundness.
#[cfg(feature = "min")]
fn recursion_proof_options() -> ProofOptions {
    ProofOptions {
        blowup_factor: 2,
        fri_number_of_queries: 1,
        coset_offset: 3,
        grinding_factor: 1,
    }
}

/// 128-bit security (multi-query).
#[cfg(feature = "blowup8")]
fn recursion_proof_options() -> ProofOptions {
    GoldilocksCubicProofOptions::with_blowup(8).expect("blowup=8 is always valid")
}

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
    let (vm_proof, inner_elf, decode_commitment, page_commitments): (
        VmProof,
        Vec<u8>,
        Commitment,
        Vec<(u64, Commitment)>,
    ) = postcard::from_bytes(&blob).expect("failed to deserialize recursion input");
    lambda_vm_prover::profile_markers::step_marker::<
        { lambda_vm_prover::profile_markers::STEP_DECODE_DONE },
    >();

    let options = recursion_proof_options();
    let ok = lambda_vm_prover::verify_with_options(
        &vm_proof,
        &inner_elf,
        &options,
        Some(decode_commitment),
        Some(&page_commitments),
    )
    .expect("verify errored");
    assert!(ok, "inner proof failed verification");

    let mut output = Vec::with_capacity(32 + decode_commitment.len() + page_commitments.len() * 40);
    output.extend_from_slice(&elf_digest(&inner_elf));
    output.extend_from_slice(&decode_commitment);
    for (page_base, commitment) in &page_commitments {
        output.extend_from_slice(&page_base.to_le_bytes());
        output.extend_from_slice(commitment);
    }
    lambda_vm_syscalls::syscalls::commit(&output);
    lambda_vm_syscalls::syscalls::sys_halt();
}
