// SCRATCH (audit, not committed): probe the XOF counter path of the official
// blake3 crate. For a fixed single-block input, the root output block at
// counter t is compress(key, block, t, block_len, flags|ROOT) — so seeking an
// OutputReader to byte position t*64 exercises the v[12]/v[13] counter split
// at arbitrary t, including t >= 2^32.
use blake3::Hasher;
use std::io::Write;

fn hex(b: &[u8]) -> String {
    b.iter().map(|x| format!("{:02x}", x)).collect()
}

fn main() {
    // 64-byte single-block input, same pattern as the Python side.
    let input: Vec<u8> = (0..64).map(|i| (i % 251) as u8).collect();
    let counters: Vec<u64> = vec![
        0, 1, 2, 0xFFFF_FFFE, 0xFFFF_FFFF, 0x1_0000_0000, 0x1_0000_0001,
        0x100_0000_0000, // 2^40
        0x8000_0000_0000, // 2^47
    ];
    let stdout = std::io::stdout();
    let mut w = std::io::BufWriter::new(stdout.lock());
    for &t in &counters {
        let mut h = Hasher::new();
        h.update(&input);
        let mut reader = h.finalize_xof();
        reader.set_position(t * 64);
        let mut out = [0u8; 64];
        reader.fill(&mut out);
        writeln!(w, "{} {}", t, hex(&out)).unwrap();
    }
}
