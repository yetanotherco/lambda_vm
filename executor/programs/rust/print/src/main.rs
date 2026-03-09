#![no_main]

use lambda_vm_syscalls::syscalls::print_string;

#[unsafe(export_name = "main")]
pub fn main() -> i32 {
    print_string("Hello, World!");
    1
}
