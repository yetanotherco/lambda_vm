//! Naive recursion guest: verifies an inner lambda-vm proof inside the VM.
//!
//! Private input layout: a 12-byte `"LVMR" + version + reserved` prefix
//! followed by an rkyv archive of `lambda_vm_prover::recursion::GuestInput`
//! `{ vm_proof, inner_elf, decode_commitment, page_commitments }` (built
//! host-side by `recursion::encode_guest_input`) — the inner program's ELF
//! bytes plus its precomputed DECODE and ELF-data-page commitments, supplied
//! instead of recomputed in-VM. The prefix 16-aligns the archive in guest
//! memory (the executor maps the payload at `PRIVATE_INPUT_START + 4`, which
//! is only 4-aligned) and tags the format so the guest rejects a wrong-format
//! blob before the unsafe access. The proof is verified **in place** via
//! `recursion::verify_and_attest_blob` — no deserialization pass, no owned
//! `VmProof`.
//!
//! The `continuation` feature swaps the monolithic proof for a multi-epoch
//! `ContinuationProof` bundle on the same wire format
//! (`recursion::ContinuationGuestInput`, built by
//! `recursion::encode_continuation_guest_input`), verified via
//! `recursion::verify_continuation_and_attest` — same trust model; the bundle
//! is verified zero-copy over the archive (no owned deserialize of the large
//! bundle), like the monolithic path.
//!
//! `ProofOptions` is fixed by exactly one preset Cargo feature
//! (`min`/`blowup2`/`blowup4`/`blowup8` — a `Preset`), not private input — an
//! attacker could otherwise pick trivially weak options and have the guest
//! accept as if a real proof had been checked.
//!
//! On success commits `program_id || inner_public_output` (a single ELF parse
//! and a single full-ELF Keccak, shared between the statement absorb and the
//! `program_id` fold). The id fold is what the consumer rebinds to a trusted
//! ELF (`check_attestation`); it is not self-enforcing here — the binding is
//! established by the consumer via `recursion::check_attestation` (a
//! host-side recompute+compare), never in-guest.
//!
//! std (not `no_std`): `build-std` provides it, prove-side code is DCE'd.
//! `#![no_main]`; inits the syscalls global allocator first thing in `main`.

#![no_main]

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

    // Zero-copy: borrow the blob straight from the mapped private-input region.
    // The 12-byte prefix puts the archive at a 16-aligned guest address, so the
    // verifier's in-place doubleword loads don't trap.
    let blob = lambda_vm_syscalls::syscalls::get_private_input_slice();
    lambda_vm_prover::profile_markers::step_marker::<
        { lambda_vm_prover::profile_markers::STEP_DECODE_DONE },
    >();

    // The guest's whole job: verify the inner proof against the supplied roots
    // and, on success, produce `program_id || inner_public_output`. The id fold
    // is what the consumer rebinds to a trusted ELF (`check_attestation`); it is
    // not self-enforcing here.
    let options = PRESET.options();

    // The `attest-commitment-id` feature selects the v2 attestation: the guest
    // never parses or hashes the inner ELF (entry point + digest are supplied,
    // the PAGE layout is reconstructed from the supplied commitments) and the
    // committed `program_id` folds only the entry point + roots, not the digest.
    #[cfg(all(not(feature = "continuation"), not(feature = "attest-commitment-id")))]
    let attestation = lambda_vm_prover::recursion::verify_and_attest_blob(blob, &options)
        .expect("verify errored")
        .expect("inner proof failed verification");

    #[cfg(all(not(feature = "continuation"), feature = "attest-commitment-id"))]
    let attestation = lambda_vm_prover::recursion::verify_and_attest_blob_v2(blob, &options)
        .expect("verify errored")
        .expect("inner proof failed verification");

    #[cfg(all(feature = "continuation", not(feature = "attest-commitment-id")))]
    let attestation =
        lambda_vm_prover::recursion::verify_continuation_and_attest(blob, &options)
            .expect("verify errored")
            .expect("inner continuation proof failed verification");

    #[cfg(all(feature = "continuation", feature = "attest-commitment-id"))]
    let attestation =
        lambda_vm_prover::recursion::verify_continuation_and_attest_v2(blob, &options)
            .expect("verify errored")
            .expect("inner continuation proof failed verification");

    lambda_vm_syscalls::syscalls::commit(&attestation);
    lambda_vm_syscalls::syscalls::sys_halt();
}
