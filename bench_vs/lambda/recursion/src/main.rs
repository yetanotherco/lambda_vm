//! Naive recursion guest: verifies an inner lambda-vm proof inside the VM.
//!
//! Private input (postcard): `(VmProof, Vec<u8>, Commitment, Vec<(u64, Commitment)>)`
//! — the inner program's ELF bytes plus its precomputed DECODE and
//! ELF-data-page commitments, supplied instead of recomputed in-VM.
//! `verify_with_options` does NOT bind the supplied roots to `inner_elf`; that
//! binding is established by folding them into `program_id` (below) and having
//! the host recompute that id and compare. That recompute is expensive, so it
//! happens once at the top level in the host, never in the guest — see
//! `program_id` in the prover's `statement` module.
//!
//! `ProofOptions` is fixed by the `min`/`blowup8` Cargo feature, not private
//! input (an attacker could otherwise pick trivially weak options and have the
//! guest accept as if a real proof had been checked).
//!
//! On success commits `program_id(inner_elf, decode_commitment,
//! page_commitments) || inner_public_output` — the program identity (a fold
//! pinning the ELF together with the roots it was verified against) plus the
//! result the inner proof attested.
//!
//! std (not `no_std`): `build-std` provides it, prove-side code is DCE'd.
//! `#![no_main]`; inits the syscalls global allocator first thing in `main`.

#![no_main]

#[cfg(feature = "blowup8")]
use lambda_vm_prover::GoldilocksCubicProofOptions;
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
        fri_final_poly_log_degree: 7,
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

    // Panic -> sys_panic; unwinding is very expensive in-guest.
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

    // program_id is not self-enforcing: a consumer must recompute it natively
    // and reject on mismatch. Commit the inner output alongside it.
    let id = lambda_vm_prover::statement::program_id_from_elf(
        &inner_elf,
        &decode_commitment,
        &page_commitments,
    )
    .expect("program_id");
    let mut output = id.to_vec();
    output.extend_from_slice(&vm_proof.public_output);
    lambda_vm_syscalls::syscalls::commit(&output);
    lambda_vm_syscalls::syscalls::sys_halt();
}
