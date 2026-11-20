// Demo guest exercising the EF zkVM IO interface (`read_input` / `write_output` /
// `write_log`).
//
// Reads the private input via the EF zero-copy `read_input` shim, then emits it
// back as the public output in TWO `write_output` calls (split in halves) to
// exercise the multi-call concatenation requirement of the EF spec.
//
// Also makes one `write_log` call. That channel is a no-op on Lambda VM, so this
// proves the symbol contract rather than any output: a portable guest calling
// `write_log` links and runs. It is the only thing that does — the symbol is
// dead-stripped from guests that never call it.
use lambda_vm_syscalls as syscalls;

/// Diagnostic text for the `write_log` call. Must be valid UTF-8 per the EF spec.
const LOG_MESSAGE: &str = "ef_io_demo: echoing private input to public output\n";

pub fn main() {
    let mut buf_ptr: *const u8 = core::ptr::null();
    let mut buf_size: usize = 0;
    unsafe {
        syscalls::ef_io::read_input(&mut buf_ptr, &mut buf_size);
        syscalls::ef_io::write_log(LOG_MESSAGE.as_ptr(), LOG_MESSAGE.len());
    }

    if buf_size > 0 {
        let half = buf_size / 2;
        unsafe {
            syscalls::ef_io::write_output(buf_ptr, half);
            syscalls::ef_io::write_output(buf_ptr.add(half), buf_size - half);
        }
    }
}
