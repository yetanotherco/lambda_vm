#![no_main]

use lambda_vm_syscalls as syscalls;

#[unsafe(export_name = "main")]
pub fn main() {
    syscalls::allocator::init_allocator();
    let input: Vec<u8> = syscalls::syscalls::get_private_input().unwrap();
    let a = input[0];
    let b = input[1];
    syscalls::syscalls::commit((a + b).to_le_bytes().as_ref());
}
