// EF IO edge case: zero-size write_output followed by a real write.
//
// Verifies that a zero-length `write_output` call doesn't corrupt state
// and the subsequent call produces correct output.
use lambda_vm_syscalls as syscalls;

pub fn main() {
    let data = b"hello";

    unsafe {
        // Zero-size write — should be a no-op.
        syscalls::ef_io::write_output(data.as_ptr(), 0);
        // Real write — should produce "hello" as public output.
        syscalls::ef_io::write_output(data.as_ptr(), data.len());
    }
}
