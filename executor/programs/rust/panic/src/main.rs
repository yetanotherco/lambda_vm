#![no_main]
use lambda_vm_syscalls as syscalls;

/// This test does not verify the output, it only prints on stdout
#[unsafe(export_name = "main")]
pub fn main() -> i32 {
    syscalls::allocator::init_allocator();
    panic!("This is a panic test");
}
