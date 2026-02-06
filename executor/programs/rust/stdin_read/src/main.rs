use lambda_vm_syscalls as syscalls;
use std::io::{self, Read};

// This test tries to read from stdin using std::io::Read
// This requires sys_read to be implemented, which Lambda VM doesn't have
// The test should fail because sys_read is not defined

pub fn main() {
    let mut buffer = [0u8; 10];

    // Try to read from stdin - this calls sys_read internally
    match io::stdin().read(&mut buffer) {
        Ok(n) => {
            // If we somehow read something, commit it
            syscalls::syscalls::commit(&buffer[..n]);
        }
        Err(_) => {
            // Reading failed - this is expected since sys_read is missing
            syscalls::syscalls::commit(&[0u8]);
        }
    }
}
