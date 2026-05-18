// EF IO edge case: no private input supplied.
//
// Calls `read_input` expecting `buf_size == 0`, then emits a hardcoded
// byte via `write_output` to prove the output path works independently.
use lambda_vm_syscalls as syscalls;

pub fn main() {
    let mut buf_ptr: *const u8 = core::ptr::null();
    let mut buf_size: usize = 0;
    unsafe {
        syscalls::ef_io::read_input(&mut buf_ptr, &mut buf_size);
    }

    // Per spec: if buf_size is 0, buf_ptr is unspecified — don't dereference it.
    assert_eq!(buf_size, 0, "expected no private input");

    // Emit a hardcoded output to verify write_output works without read_input data.
    let output = b"ok";
    unsafe {
        syscalls::ef_io::write_output(output.as_ptr(), output.len());
    }
}
