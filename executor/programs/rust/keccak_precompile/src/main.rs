use lambda_vm_syscalls as syscalls;

pub fn main() {
    let input_a = b"hello world";
    let input_b = b"!";
    let mut input = [0u8; 12];
    input[..input_a.len()].copy_from_slice(input_a);
    input[input_a.len()..].copy_from_slice(input_b);

    let output = syscalls::keccak::keccak256(&input);
    syscalls::syscalls::commit(&output);
}
