use getrandom::Error;

#[unsafe(no_mangle)]
unsafe extern "Rust" fn __getrandom_v03_custom(
    _dest_ptr: *mut u8,
    _len: usize,
) -> Result<(), Error> {
    panic!("getrandom is not supported");
}
