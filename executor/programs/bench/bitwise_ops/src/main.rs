use lambda_vm_syscalls as syscalls;

const ITERATIONS: usize = {
    const fn parse(s: &str) -> usize {
        let b = s.as_bytes();
        let mut r = 0;
        let mut i = 0;
        while i < b.len() {
            assert!(b[i] >= b'0' && b[i] <= b'9');
            r = r * 10 + (b[i] - b'0') as usize;
            i += 1;
        }
        r
    }
    match option_env!("ITERATIONS") {
        Some(s) => parse(s),
        None => 286000,
    }
};

#[inline(never)]
fn bitwise_mix(a: u32, b: u32) -> u32 {
    let x = a ^ b;
    let y = (a & b) | (!a & !b);
    let z = (a >> 16) | (b << 16);
    x.wrapping_add(y).wrapping_add(z)
}

pub fn main() {
    let mut result = 0x12345678u32;
    let mut i = 0;

    while i < ITERATIONS {
        result = bitwise_mix(result, i as u32);
        result = result.rotate_left(5);
        result ^= i as u32;
        i += 1;
    }

    syscalls::syscalls::commit(&result.to_le_bytes());
}
