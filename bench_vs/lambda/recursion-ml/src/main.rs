#![no_main]
//! Verifies multilinear continuation epochs in the guest. The private input is
//! `elf_len (u64 LE) || elf || rkyv(Vec<EpochProof>)`; commits 1 on success.

use lambda_vm_prover::multilinear_continuation::{EpochProof, verify_epochs};

#[unsafe(export_name = "main")]
pub fn main() -> ! {
    lambda_vm_syscalls::allocator::init_allocator();
    const PANIC_MSG: &str = "PANICKED";
    std::panic::set_hook(Box::new(|_| unsafe {
        lambda_vm_syscalls::syscalls::sys_panic(PANIC_MSG.as_ptr(), PANIC_MSG.len())
    }));

    let blob = lambda_vm_syscalls::syscalls::get_private_input_slice();
    let elf_len = u64::from_le_bytes(blob[..8].try_into().unwrap()) as usize;
    let elf = &blob[8..8 + elf_len];
    let mut aligned = rkyv::util::AlignedVec::<16>::new();
    aligned.extend_from_slice(&blob[8 + elf_len..]);
    let epochs: Vec<EpochProof> =
        rkyv::from_bytes::<Vec<EpochProof>, rkyv::rancor::Error>(&aligned).expect("decode");

    let opts = lambda_vm_prover::GoldilocksCubicProofOptions::with_blowup(2).expect("opts");
    let ok = verify_epochs(elf, &epochs, &opts).expect("verify errored");
    assert!(ok, "the epochs do not verify");
    lambda_vm_syscalls::syscalls::commit(&[ok as u8]);
    lambda_vm_syscalls::syscalls::sys_halt();
}
