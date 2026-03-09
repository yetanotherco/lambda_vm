#![no_main]

use std::io::stdout;
use std::io::Write;
use lambda_vm_syscalls as syscalls;

/// This test does not verify the output, it only prints on stdout
#[unsafe(export_name = "main")]
pub fn main() -> i32 {
    syscalls::allocator::init_allocator();
    stdout().write(b"Hello from sys_write!\n").unwrap();
    return 1;
}
