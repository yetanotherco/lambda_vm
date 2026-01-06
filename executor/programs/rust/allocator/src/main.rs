#![no_main]

extern crate alloc;
use alloc::string::String;
use lambda_vm_syscalls as syscalls;

#[unsafe(export_name = "main")]
pub fn main() -> u32 {
    syscalls::allocator::init_allocator();
    let hello = String::from("Hello World");
    return hello.len() as u32;
}
