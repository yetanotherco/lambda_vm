use lambda_vm_syscalls as syscalls;

const NUM_ITERATIONS: usize = 20600;

#[inline(never)]
fn mod_exp(base: u32, exp: u32, modulus: u32) -> u32 {
    if modulus == 1 {
        return 0;
    }

    let mut result = 1u64;
    let mut base = (base as u64) % (modulus as u64);
    let mut exp = exp;
    let modulus = modulus as u64;

    while exp > 0 {
        if exp & 1 == 1 {
            result = (result * base) % modulus;
        }
        exp >>= 1;
        base = (base * base) % modulus;
    }

    result as u32
}

pub fn main() {
    let mut checksum = 0u32;
    let mut base = 2u32;
    let mut exp = 1000u32;
    let modulus = 1000000007u32;

    let mut i = 0;
    while i < NUM_ITERATIONS {
        let result = mod_exp(base, exp, modulus);
        checksum = checksum.wrapping_add(result);

        base = base.wrapping_mul(3).wrapping_add(1) % 1000;
        if base == 0 {
            base = 2;
        }
        exp = exp.wrapping_add(7) % 10000;
        if exp == 0 {
            exp = 1;
        }

        i += 1;
    }

    syscalls::syscalls::commit(&checksum.to_le_bytes());
}
