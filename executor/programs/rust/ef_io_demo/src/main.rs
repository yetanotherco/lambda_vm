// Demo guest exercising the EF zkVM IO interface (`read_input` / `write_output`).
//
// Reads the private input via the EF zero-copy `read_input` shim, then emits it
// back as the public output in TWO `write_output` calls (split in halves) to
// exercise the multi-call concatenation requirement of the EF spec.
use lambda_vm_syscalls as syscalls;

pub fn main() {
    let mut buf_ptr: *const u8 = core::ptr::null();
    let mut buf_size: usize = 0;
    unsafe {
        syscalls::ef_io::read_input(&mut buf_ptr, &mut buf_size);
    }

    if buf_size > 0 {
        let half = buf_size / 2;
        unsafe {
            syscalls::ef_io::write_output(buf_ptr, half);
            syscalls::ef_io::write_output(buf_ptr.add(half), buf_size - half);
        }
    }
}
