#![no_main]
ziskos::entrypoint!(main);

fn main() {
    let n: u64 = ziskos::io::read();
    let mut a: u64 = 0;
    let mut b: u64 = 1;
    for _ in 0..n {
        let c = a.wrapping_add(b);
        a = b;
        b = c;
    }
    ziskos::io::commit(&b);
}
