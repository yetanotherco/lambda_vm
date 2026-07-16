use lambda_vm_syscalls::keccak::keccak256;
use lambda_vm_syscalls::syscalls;

// Exercises the `keccak_permute`-ecall-backed sponge (`lambda_vm_syscalls::keccak`)
// against known Keccak-256 vectors: empty input, one rate block minus one byte,
// exactly one rate block, and multi-block input — the padding edge cases a
// single small input can't cover.
pub fn main() {
    const RATE_BYTES: usize = 136;

    let empty = keccak256(b"");
    let abc = keccak256(b"abc");
    let rate_minus_one = keccak256(&[0x5a; RATE_BYTES - 1]);
    let exactly_rate = keccak256(&[0x3c; RATE_BYTES]);
    let multi_block_input: Vec<u8> = (0..2 * RATE_BYTES + 17).map(|i| i as u8).collect();
    let multi_block = keccak256(&multi_block_input);

    let mut output = Vec::with_capacity(5 * 32);
    output.extend_from_slice(&empty);
    output.extend_from_slice(&abc);
    output.extend_from_slice(&rate_minus_one);
    output.extend_from_slice(&exactly_rate);
    output.extend_from_slice(&multi_block);

    syscalls::commit(&output);
}
