#![no_main]
sp1_zkvm::entrypoint!(main);

pub fn main() {
    let n: u64 = sp1_zkvm::io::read::<u64>();
    let mut a: u64 = 0;
    let mut b: u64 = 1;
    for _ in 0..n {
        let c = a.wrapping_add(b);
        a = b;
        b = c;
    }
    sp1_zkvm::io::commit(&b);
}
