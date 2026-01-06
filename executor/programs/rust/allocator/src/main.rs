#![no_main]

extern crate alloc;
use alloc::string::String;
use lambda_vm_syscalls as syscalls;

#[unsafe(export_name = "main")]
pub fn main() {
    syscalls::allocator::init_allocator();
    let hello = String::from("Hello World");
    syscalls::syscalls::commit(hello.as_bytes());
}
