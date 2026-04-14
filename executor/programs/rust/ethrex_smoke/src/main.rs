use guest_program::input::ProgramInput;
use lambda_vm_syscalls as syscalls;
use rkyv::rancor::Error;

pub fn main() {
    let raw = syscalls::syscalls::get_private_input().unwrap();
    let input = rkyv::from_bytes::<ProgramInput, Error>(&raw).unwrap();
    // Commit block count as proof that deserialization worked
    let count = input.blocks.len() as u32;
    syscalls::syscalls::commit(&count.to_le_bytes());
}
