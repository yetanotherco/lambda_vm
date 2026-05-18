// EF IO edge case: many small write_output calls.
//
// Emits "abcdefghij" one byte at a time via 10 separate `write_output`
// calls to stress multi-call concatenation beyond the two-call demo.
use lambda_vm_syscalls as syscalls;

pub fn main() {
    let data = b"abcdefghij";
    for &byte in data.iter() {
        unsafe {
            syscalls::ef_io::write_output(&byte as *const u8, 1);
        }
    }
}
