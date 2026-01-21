use lambda_vm_syscalls as syscalls;
use std::array;

const MAX_COMMIT_SIZE: usize = 1024;
const ITERATIONS: usize = 9000;

pub fn main() {
    for n in 0..ITERATIONS {
        // Commit a string
        let string = format!("Hello World [{n}]");
        syscalls::syscalls::commit(&string.as_bytes());
        // Commit a medium-length byte array
        let large_byte_array: [u8; MAX_COMMIT_SIZE / 2] =
            array::from_fn(|i| n as u8 * i as u8 + 32);
        syscalls::syscalls::commit(&large_byte_array);
        // Commit a large byte array
        let large_byte_array: [u8; MAX_COMMIT_SIZE] = array::from_fn(|i| n as u8 * i as u8 + 16);
        syscalls::syscalls::commit(&large_byte_array);
    }
}
