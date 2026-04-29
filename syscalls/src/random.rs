use std::sync::Mutex;

use getrandom::Error;
use lazy_static::lazy_static;
use rand::Rng;
use rand::{SeedableRng, rngs::StdRng};

const RANDOM_SEED: u64 = 0x1234567890abcdef;

lazy_static! {
    static ref RNG: Mutex<StdRng> = Mutex::new(StdRng::seed_from_u64(RANDOM_SEED));
}

/// # Safety
///
/// This function should not be called by the user
/// It is only for rust std internal uses
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sys_rand(buf: *mut u8, len: usize) {
    // NOTE: no print_string here — the Print ecall (syscall 1) has no
    // receiver on the Ecall bus and would cause a verification failure.
    let mut rng = RNG.lock().unwrap();
    for i in 0..len {
        unsafe {
            let element = buf.add(i);
            *element = rng.random();
        }
    }
}

#[unsafe(no_mangle)]
unsafe extern "Rust" fn __getrandom_v03_custom(
    _dest_ptr: *mut u8,
    _len: usize,
) -> Result<(), Error> {
    panic!("getrandom is not supported");
}
