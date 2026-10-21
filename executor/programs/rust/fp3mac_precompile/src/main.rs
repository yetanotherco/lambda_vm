use lambda_vm_syscalls::syscalls;

/// Cells the Horner runs over.
const CELLS: usize = 48;

// Exercises the cubic-extension multiply-add ecall: a Horner evaluation
// `acc ← acc·al + x` over pseudo-random canonical cells (edge values first),
// committing the three words of `acc`.
pub fn main() {
    const P: u64 = 0xFFFF_FFFF_0000_0001;
    let mut seed = 0x9E37_79B9_7F4A_7C15u64;
    let mut next = || {
        seed = seed.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = seed;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        (z ^ (z >> 31)) % P
    };
    let mut state: [u64; 6] = [0, 0, 0, next(), next(), next()];
    let edges: [[u64; 3]; 3] = [[0, 1, P - 1], [P - 1, P - 1, P - 1], [0xFFFF_FFFF, 0, 1]];
    for k in 0..CELLS {
        let x = if k < edges.len() {
            edges[k]
        } else {
            [next(), next(), next()]
        };
        syscalls::fp3mac(&mut state, &x);
    }
    let mut output = Vec::with_capacity(3 * 8);
    for w in &state[..3] {
        output.extend_from_slice(&w.to_le_bytes());
    }
    syscalls::commit(&output);
}
