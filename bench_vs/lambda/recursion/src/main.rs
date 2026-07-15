//! Naive recursion guest: verifies an inner lambda-vm proof inside the VM.
//!
//! Private input (postcard): `lambda_vm_prover::recursion::GuestInput` — the
//! inner proof, the inner program's ELF bytes, and its precomputed DECODE and
//! ELF-data-page commitments, supplied instead of recomputed in-VM.
//!
//! `ProofOptions` is fixed by exactly one preset Cargo feature
//! (`min`/`blowup2`/`blowup4`/`blowup8` — a `Preset`), not private input — an
//! attacker could otherwise pick trivially weak options and have the guest
//! accept as if a real proof had been checked.
//!
//! On success commits `program_id || inner_public_output` via
//! `recursion::verify_and_attest` (a single ELF parse and a single full-ELF
//! Keccak, shared between the statement absorb and the `program_id` fold). The
//! attestation is not self-enforcing: the binding is established by the
//! consumer via `recursion::check_attestation` (a host-side recompute+compare),
//! never in-guest.
//!
//! std (not `no_std`): `build-std` provides it, prove-side code is DCE'd.
//! `#![no_main]`; inits the syscalls global allocator first thing in `main`.

#![no_main]

#[cfg(not(feature = "continuation"))]
use lambda_vm_prover::recursion::GuestInput;
use lambda_vm_prover::recursion::Preset;

#[cfg(not(any(
    feature = "min",
    feature = "blowup2",
    feature = "blowup4",
    feature = "blowup8"
)))]
compile_error!("select exactly one of the `min`/`blowup2`/`blowup4`/`blowup8` features");
#[cfg(any(
    all(feature = "min", feature = "blowup2"),
    all(feature = "min", feature = "blowup4"),
    all(feature = "min", feature = "blowup8"),
    all(feature = "blowup2", feature = "blowup4"),
    all(feature = "blowup2", feature = "blowup8"),
    all(feature = "blowup4", feature = "blowup8"),
))]
compile_error!("select exactly one of the `min`/`blowup2`/`blowup4`/`blowup8` features");

/// The build preset fixing the inner `ProofOptions` (see the module docs).
#[cfg(feature = "min")]
const PRESET: Preset = Preset::Min;
#[cfg(feature = "blowup2")]
const PRESET: Preset = Preset::Blowup2;
#[cfg(feature = "blowup4")]
const PRESET: Preset = Preset::Blowup4;
#[cfg(feature = "blowup8")]
const PRESET: Preset = Preset::Blowup8;

#[unsafe(export_name = "main")]
pub fn main() -> ! {
    lambda_vm_syscalls::allocator::init_allocator();

    // Panic -> sys_panic; unwinding is very expensive in-guest.
    const PANIC_MSG: &str = "PANICKED";
    std::panic::set_hook(Box::new(|_| unsafe {
        lambda_vm_syscalls::syscalls::sys_panic(PANIC_MSG.as_ptr(), PANIC_MSG.len())
    }));

    let blob = lambda_vm_syscalls::syscalls::get_private_input();

    // The guest's whole job: verify the inner proof against the supplied roots
    // and, on success, produce `program_id || inner_public_output`. The id fold
    // is what the consumer rebinds to a trusted ELF (`check_attestation`); it is
    // not self-enforcing here. The `continuation` feature swaps the monolithic
    // VmProof for a multi-epoch ContinuationProof bundle; same trust model.
    let options = PRESET.options();

    #[cfg(not(feature = "continuation"))]
    let attestation = {
        let (vm_proof, inner_elf, decode_commitment, page_commitments): GuestInput =
            postcard::from_bytes(&blob).expect("failed to deserialize recursion input");
        lambda_vm_prover::profile_markers::step_marker::<
            { lambda_vm_prover::profile_markers::STEP_DECODE_DONE },
        >();
        lambda_vm_prover::recursion::verify_and_attest(
            &vm_proof,
            &inner_elf,
            &options,
            decode_commitment,
            &page_commitments,
        )
        .expect("verify errored")
        .expect("inner proof failed verification")
    };

    #[cfg(feature = "continuation")]
    let attestation = {
        let (bundle, inner_elf, decode_commitment, page_commitments): lambda_vm_prover::recursion::ContinuationGuestInput =
            postcard::from_bytes(&blob).expect("failed to deserialize recursion input");
        lambda_vm_prover::profile_markers::step_marker::<
            { lambda_vm_prover::profile_markers::STEP_DECODE_DONE },
        >();
        lambda_vm_prover::recursion::verify_continuation_and_attest(
            &bundle,
            &inner_elf,
            &options,
            decode_commitment,
            &page_commitments,
        )
        .expect("verify errored")
        .expect("inner continuation proof failed verification")
    };

    lambda_vm_syscalls::syscalls::commit(&attestation);
    lambda_vm_syscalls::syscalls::sys_halt();
}
