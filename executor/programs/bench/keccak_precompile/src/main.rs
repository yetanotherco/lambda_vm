use lambda_vm_syscalls as syscalls;

const ITERATIONS: usize = 1000;

pub fn main() {
    let mut output = [0u8; 32];

    for _ in 0..ITERATIONS {
        output = syscalls::keccak::keccak256(&output);
    }

    syscalls::syscalls::commit(&output);
}
