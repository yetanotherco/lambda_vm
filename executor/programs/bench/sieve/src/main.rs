#![no_std]
#![no_main]

use core::panic::PanicInfo;

#[panic_handler]
fn panic(_info: &PanicInfo) -> ! {
    loop {}
}

const LIMIT: usize = 37000;

#[inline(never)]
fn sieve(is_prime: &mut [bool; LIMIT + 1]) -> u32 {
    // Initialize
    let mut i = 0;
    while i <= LIMIT {
        is_prime[i] = true;
        i += 1;
    }
    is_prime[0] = false;
    is_prime[1] = false;

    // Sieve
    let mut i = 2;
    while i * i <= LIMIT {
        if is_prime[i] {
            let mut j = i * i;
            while j <= LIMIT {
                is_prime[j] = false;
                j += i;
            }
        }
        i += 1;
    }

    // Count primes
    let mut count = 0u32;
    let mut k = 0;
    while k <= LIMIT {
        if is_prime[k] {
            count += 1;
        }
        k += 1;
    }
    count
}

#[unsafe(no_mangle)]
pub fn main() -> u32 {
    let mut is_prime = [false; LIMIT + 1];
    // Primes up to 10000: there are 1229 primes
    sieve(&mut is_prime)
}
