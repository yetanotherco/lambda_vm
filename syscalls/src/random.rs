use std::sync::Mutex;

use getrandom::Error;
use lazy_static::lazy_static;
use rand::Rng;
use rand::{RngCore, SeedableRng, rngs::StdRng};

use crate::syscalls::print_string;

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
    print_string("sys_rand called\n");
    print_string("WARNING: Using sys_rand is insecure\n");

    let mut rng = RNG.lock().unwrap();
    for i in 0..len {
        unsafe {
            let element = buf.add(i);
            *element = rng.random();
        }
    }
}

/// Custom getrandom v0.3 backend (selected via `--cfg getrandom_backend="custom"`).
///
/// Fills `dest` with deterministic bytes from the constant-seeded `StdRng` (ChaCha20)
/// instead of panicking. This keeps weak-random consumers (e.g. `std::HashMap`'s
/// `RandomState`) working in-guest at the cost of being insecure — the seed is fixed.
///
/// # Safety
///
/// `dest_ptr` must be valid for writes of `len` bytes.
#[unsafe(no_mangle)]
unsafe extern "Rust" fn __getrandom_v03_custom(
    dest_ptr: *mut u8,
    len: usize,
) -> Result<(), Error> {
    print_string("getrandom called\n");
    print_string("WARNING: Using getrandom is insecure\n");

    let mut rng = RNG.lock().unwrap();
    let dest = unsafe { core::slice::from_raw_parts_mut(dest_ptr, len) };
    rng.fill_bytes(dest);
    Ok(())
}
