use lambda_vm_syscalls::syscalls;

// Exercises the width-16 Poseidon1 ecall: a chain of permutations from a
// state with edge values (0, 1, p − 1), committing every lane of the last.
pub fn main() {
    const P: u64 = 0xFFFF_FFFF_0000_0001;
    let mut state: [u64; 16] = core::array::from_fn(|i| match i % 4 {
        0 => 0,
        1 => 1,
        2 => P - 1,
        _ => (i as u64) * 0x0123_4567_89AB_CDEF % P,
    });
    for _ in 0..5 {
        syscalls::p1w16_permute(&mut state);
    }
    let mut output = Vec::with_capacity(16 * 8);
    for lane in state {
        output.extend_from_slice(&lane.to_le_bytes());
    }
    syscalls::commit(&output);
}
