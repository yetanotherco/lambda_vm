// Ground-truth BLAKE3 vector generator using the OFFICIAL blake3 crate (v1.8.5,
// pure-Rust feature, built offline from the local cargo registry).
// Emits JSON on stdout in the same shape as the upstream test_vectors.json,
// plus a randomised differential set.

use blake3::Hasher;
use std::io::Write;

const KEY: &[u8; 32] = b"whats the Elvish word for friend";
const CONTEXT: &str = "BLAKE3 2019-12-27 16:29:52 test vectors context";
const XOF_LEN: usize = 131;

fn pattern_input(n: usize) -> Vec<u8> {
    (0..n).map(|i| (i % 251) as u8).collect()
}

fn hash_hex(input: &[u8], out_len: usize) -> String {
    let mut h = Hasher::new();
    h.update(input);
    let mut out = vec![0u8; out_len];
    h.finalize_xof().fill(&mut out);
    hex(&out)
}

fn keyed_hex(key: &[u8; 32], input: &[u8], out_len: usize) -> String {
    let mut h = Hasher::new_keyed(key);
    h.update(input);
    let mut out = vec![0u8; out_len];
    h.finalize_xof().fill(&mut out);
    hex(&out)
}

fn derive_hex(ctx: &str, input: &[u8], out_len: usize) -> String {
    let mut h = Hasher::new_derive_key(ctx);
    h.update(input);
    let mut out = vec![0u8; out_len];
    h.finalize_xof().fill(&mut out);
    hex(&out)
}

fn hex(b: &[u8]) -> String {
    b.iter().map(|x| format!("{:02x}", x)).collect()
}

// xorshift64* — deterministic, self-contained RNG so the Python side can
// reproduce the exact same inputs without sharing any code.
struct Rng(u64);
impl Rng {
    fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        x.wrapping_mul(0x2545F4914F6CDD1D)
    }
    fn byte(&mut self) -> u8 {
        (self.next_u64() >> 33) as u8
    }
}

fn main() {
    let lengths: Vec<usize> = vec![
        0, 1, 2, 3, 4, 5, 6, 7, 8, 63, 64, 65, 127, 128, 129, 1023, 1024, 1025, 2048, 2049, 3072,
        3073, 4096, 4097, 5120, 5121, 6144, 6145, 7168, 7169, 8192, 8193, 16384, 31744, 102400,
    ];

    let stdout = std::io::stdout();
    let mut w = std::io::BufWriter::new(stdout.lock());

    writeln!(w, "{{").unwrap();
    writeln!(w, "  \"key\": \"{}\",", String::from_utf8_lossy(KEY)).unwrap();
    writeln!(w, "  \"context_string\": \"{}\",", CONTEXT).unwrap();
    writeln!(w, "  \"cases\": [").unwrap();
    for (i, &n) in lengths.iter().enumerate() {
        let inp = pattern_input(n);
        writeln!(w, "    {{").unwrap();
        writeln!(w, "      \"input_len\": {},", n).unwrap();
        writeln!(w, "      \"hash\": \"{}\",", hash_hex(&inp, XOF_LEN)).unwrap();
        writeln!(w, "      \"keyed_hash\": \"{}\",", keyed_hex(KEY, &inp, XOF_LEN)).unwrap();
        writeln!(w, "      \"derive_key\": \"{}\"", derive_hex(CONTEXT, &inp, XOF_LEN)).unwrap();
        writeln!(w, "    }}{}", if i + 1 == lengths.len() { "" } else { "," }).unwrap();
    }
    writeln!(w, "  ],").unwrap();

    // Randomised differential set. Inputs are generated from a self-contained
    // xorshift64* stream that the Python side re-implements independently.
    writeln!(w, "  \"random\": [").unwrap();
    let rlens: Vec<usize> = vec![
        0, 1, 2, 31, 32, 33, 63, 64, 65, 127, 128, 129, 512, 1000, 1023, 1024, 1025, 2048, 4096,
        4097, 10000, 65536, 100000,
    ];
    let xofs: Vec<usize> = vec![16, 32, 64, 131, 200];
    let mut seedctr: u64 = 1;
    let mut first = true;
    for &n in &rlens {
        for &xl in &xofs {
            let seed = seedctr;
            seedctr += 1;
            let mut rng = Rng(seed);
            let msg: Vec<u8> = (0..n).map(|_| rng.byte()).collect();
            let mut krng = Rng(seed ^ 0xDEADBEEF);
            let mut key = [0u8; 32];
            for b in key.iter_mut() {
                *b = krng.byte();
            }
            let ctx = format!("lambda-vm oracle review ctx {}/{}", n, xl);
            if !first {
                writeln!(w, ",").unwrap();
            }
            first = false;
            write!(
                w,
                "    {{\"seed\": {}, \"len\": {}, \"xof\": {}, \"key\": \"{}\", \"ctx\": \"{}\", \"hash\": \"{}\", \"keyed\": \"{}\", \"derive\": \"{}\"}}",
                seed,
                n,
                xl,
                hex(&key),
                ctx,
                hash_hex(&msg, xl),
                keyed_hex(&key, &msg, xl),
                derive_hex(&ctx, &msg, xl)
            )
            .unwrap();
        }
    }
    writeln!(w, "\n  ],").unwrap();

    // A couple of well-known digests, for a human sanity check.
    writeln!(
        w,
        "  \"known\": {{\"empty\": \"{}\", \"abc\": \"{}\"}}",
        hash_hex(b"", 32),
        hash_hex(b"abc", 32)
    )
    .unwrap();
    writeln!(w, "}}").unwrap();
}
