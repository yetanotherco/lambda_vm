//! SHA-256 compression only. Padding and the IV belong to guest code.
//! The syscall ABI uses big-endian bytes for both the state and message.
pub const K: [u32; 64] = [
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2,
];
pub const IV: [u32; 8] = [
    0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19,
];
pub fn sigma(x: u32, kind: usize) -> u32 {
    match kind {
        0 => x.rotate_right(7) ^ x.rotate_right(18) ^ (x >> 3),
        1 => x.rotate_right(17) ^ x.rotate_right(19) ^ (x >> 10),
        2 => x.rotate_right(2) ^ x.rotate_right(13) ^ x.rotate_right(22),
        3 => x.rotate_right(6) ^ x.rotate_right(11) ^ x.rotate_right(25),
        _ => unreachable!(),
    }
}
pub fn schedule(m: &[u8; 64]) -> [u32; 64] {
    let mut w = [0u32; 64];
    for i in 0..16 {
        w[i] = u32::from_be_bytes(m[4 * i..4 * i + 4].try_into().unwrap());
    }
    for i in 16..64 {
        w[i] = w[i - 16]
            .wrapping_add(sigma(w[i - 15], 0))
            .wrapping_add(w[i - 7])
            .wrapping_add(sigma(w[i - 2], 1));
    }
    w
}
pub fn round(s: [u32; 8], w: u32, k: u32) -> [u32; 8] {
    let [a, b, c, d, e, f, g, h] = s;
    let t1 = h
        .wrapping_add(sigma(e, 3))
        .wrapping_add((e & f) ^ (!e & g))
        .wrapping_add(k)
        .wrapping_add(w);
    let t2 = sigma(a, 2).wrapping_add((a & b) ^ (a & c) ^ (b & c));
    [t1.wrapping_add(t2), a, b, c, d.wrapping_add(t1), e, f, g]
}
pub fn compress(h: &mut [u8; 32], m: &[u8; 64]) {
    let initial =
        std::array::from_fn(|i| u32::from_be_bytes(h[i * 4..i * 4 + 4].try_into().unwrap()));
    let mut s = initial;
    let w = schedule(m);
    for i in 0..64 {
        s = round(s, w[i], K[i]);
    }
    for i in 0..8 {
        h[4 * i..4 * i + 4].copy_from_slice(&s[i].wrapping_add(initial[i]).to_be_bytes());
    }
}
