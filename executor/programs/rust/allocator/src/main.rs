extern crate alloc;
use alloc::string::String;
use lambda_vm_syscalls as syscalls;

pub fn main() {
    let hello = String::from("Hello World");
    syscalls::syscalls::commit(hello.as_bytes());
}
