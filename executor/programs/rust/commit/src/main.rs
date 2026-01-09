#![no_main]

use lambda_vm_syscalls as syscalls;

#[unsafe(export_name = "main")]
pub fn main() {
    syscalls::allocator::init_allocator();
    let input: Vec<u8> = syscalls::syscalls::get_private_input().unwrap();
    syscalls::syscalls::print_string(format!("Private input received: {:?}\n", input).as_str());
    syscalls::syscalls::commit(&input);
}
