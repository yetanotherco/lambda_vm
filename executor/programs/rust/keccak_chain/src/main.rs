use lambda_vm_syscalls::keccak::keccak256;
use lambda_vm_syscalls::syscalls;

// N chained Merkle compressions, h = keccak256(h || h): one keccak-f each plus
// the byte handling a RISC-V verifier does around it. N is the first four
// bytes of the private input, little-endian.
pub fn main() {
    let input = syscalls::get_private_input();
    let n = u32::from_le_bytes(input[..4].try_into().expect("four bytes of N"));
    let mut h = [7u8; 32];
    let mut pair = [0u8; 64];
    for _ in 0..n {
        pair[..32].copy_from_slice(&h);
        pair[32..].copy_from_slice(&h);
        h = keccak256(&pair);
    }
    syscalls::commit(&h);
}
