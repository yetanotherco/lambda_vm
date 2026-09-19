//! A workload whose SHA-256 sub-trace is the same shape as the ethrex block's,
//! with everything else stripped away.
//!
//! 140 hashes of a 1024-byte input are 140 * 17 = 2380 compression blocks,
//! which is exactly what ethrex block 25453112 performs. The five SHA chips
//! therefore get the same row counts — and so the same padding regime — while
//! the rest of the trace shrinks from ~2.9e9 cells to almost nothing. That
//! makes a change worth 1% of the ethrex proof worth tens of percent here,
//! which is the difference between a measurable effect and one inside CV.
const CALLS: usize = 140;
const LEN: usize = 1024;

fn main() {
    let mut data = [0u8; LEN];
    for (i, b) in data.iter_mut().enumerate() {
        *b = (i * 31 + 7) as u8;
    }
    let mut acc = [0u8; 32];
    for c in 0..CALLS {
        data[0] = c as u8;
        let hash = lambda_vm_syscalls::sha256::sha256(&data);
        for (a, h) in acc.iter_mut().zip(hash.iter()) {
            *a ^= h;
        }
    }
    lambda_vm_syscalls::syscalls::commit(&acc);
}
