use lambda_vm_syscalls as syscalls;
use tiny_keccak::Hasher;

pub fn main() {
    let input_a = b"hello world";
    let input_b = b"!";
    let mut output = [0u8; 32];
    let mut hasher = tiny_keccak::Keccak::v256();
    hasher.update(input_a);
    hasher.update(input_b);
    hasher.finalize(&mut output);
    syscalls::syscalls::commit(&output);
}
