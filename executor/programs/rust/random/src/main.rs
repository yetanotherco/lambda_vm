#![no_main]
use lambda_vm_syscalls as syscalls;

#[unsafe(export_name = "main")]
pub fn main() {
    syscalls::allocator::init_allocator();
    let x = rand::random::<u8>();
    syscalls::syscalls::commit(x.to_le_bytes().as_ref());
}
