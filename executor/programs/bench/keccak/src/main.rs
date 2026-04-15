use lambda_vm_syscalls as syscalls;
use tiny_keccak::Hasher;

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
        None => 1000,
    }
};

pub fn main() {
    let mut output = [0u8; 32];

    for _ in 0..ITERATIONS {
        let mut hasher = tiny_keccak::Keccak::v256();
        hasher.update(&output);
        hasher.finalize(&mut output);
    }

    syscalls::syscalls::commit(&output);
}
