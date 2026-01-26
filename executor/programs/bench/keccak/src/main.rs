use lambda_vm_syscalls as syscalls;
use tiny_keccak::Hasher;

const ITERATIONS: usize = 1000;

pub fn main() {
    let mut output = [0u8; 32];

    for _ in 0..ITERATIONS {
        let mut hasher = tiny_keccak::Keccak::v256();
        hasher.update(&output);
        hasher.finalize(&mut output);
    }

    syscalls::syscalls::commit(&output);
}
