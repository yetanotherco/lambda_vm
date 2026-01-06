#![no_main]

use lambda_vm_syscalls as syscalls;

#[unsafe(export_name = "main")]
pub fn main() -> i32 {
    syscalls::allocator::init_allocator();
    let vector = vec![1, 2, 3, 4, 5];
    return vector.iter().sum();
}
