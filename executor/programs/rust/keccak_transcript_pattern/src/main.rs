use crypto::hash::platform_keccak::PlatformKeccak256;
use digest::Digest;
use lambda_vm_syscalls::syscalls;

// Exercises `PlatformKeccak256` (the `keccak_permute`-ecall-backed sponge used
// by `DefaultTranscript`) with the same call pattern `DefaultTranscript::sample`
// drives: several small, non-rate-aligned `update()`s, then `finalize_reset()`,
// then more `update()`s seeded with the reversed prior digest. This covers the
// cross-call buffering path that a single one-shot hash can't reach.
pub fn main() {
    let mut hasher = PlatformKeccak256::new();
    hasher.update(&[0xaa; 5]);
    hasher.update(&[0xbb; 40]);
    hasher.update(&[0xcc; 17]);
    hasher.update(&[0xdd; 100]);
    let digest1: [u8; 32] = hasher.finalize_reset().into();

    let mut reversed1 = digest1;
    reversed1.reverse();
    hasher.update(&reversed1);
    hasher.update(&[0xee; 3]);
    hasher.update(&[0xff; 130]);
    let digest2: [u8; 32] = hasher.finalize_reset().into();

    let mut reversed2 = digest2;
    reversed2.reverse();
    hasher.update(&reversed2);
    let digest3: [u8; 32] = hasher.finalize().into();

    let mut output = Vec::with_capacity(3 * 32);
    output.extend_from_slice(&digest1);
    output.extend_from_slice(&digest2);
    output.extend_from_slice(&digest3);
    syscalls::commit(&output);
}
