//! The BLAKE3 **6-round internal variant** compression function, vendored into
//! the LFM tree from PR #903 (`yetanotherco/lambda_vm`, head
//! `89aeeb8c2b0389e9d21a861c9e3a10a7b1b5704e`).
//!
//! # Why a copy, and why here
//!
//! PR #903 lands this primitive in `executor/src/vm/instruction/execution.rs`
//! and its chip in `prover/src/tables/blake3.rs` — both production paths this
//! branch may not touch. The hash-matrix leg needs the primitive to *measure* a
//! candidate column, not to ship it, so the port lives under `lfm/` where it is
//! additive by construction. When #903 merges, this module should be deleted
//! and the executor's `blake3_compress_6round` used directly; the vectors below
//! are the check that the two agree.
//!
//! # Provenance of the primitive, and why no external KAT exists (rule 9)
//!
//! Standing-decisions rule 9 requires pinning a new primitive against an
//! external known-answer vector that nothing in this repository produced. That
//! is *impossible in the usual form* for this hash: the 6-round variant is not
//! standard BLAKE3 (7 rounds), so no published vector and no crate exposes it.
//! The provenance chain #903 supplies instead, and which this module inherits:
//!
//! 1. A z3-proved model of the compression dataflow
//!    (`thoughts/blake3/blake3-chip/z3_blake_verify.py`).
//! 2. A Python oracle (`thoughts/blake3/blake3-oracle/blake3_ref.py`) whose
//!    **7-round** instantiation is pinned against the official `blake3` crate's
//!    published test vectors (`official_test_vectors.json`) — so the oracle's
//!    G-function, message schedule, counter split and feed-forward are all
//!    externally validated; only the round count is varied.
//! 3. That oracle at `rounds = 6` emitted the 10 canonical vectors in
//!    [`CANONICAL_VECTORS`], which pin this port.
//!
//! So the external anchor is one step removed: the *conventions* are pinned by
//! the official crate through the oracle, and the round count is the single
//! degree of freedom the canonical vectors add. That is weaker than a direct
//! KAT and is recorded as such — but [`CANONICAL_VECTORS`] still discriminates
//! every convention a wrong port could get wrong, which the falsification tests
//! at the bottom of this file demonstrate one convention at a time.
//!
//! ⚠ Security assumption **A6R**: collision resistance of the 6-round variant
//! is a named, unratified assumption (#903's `IMPLEMENTATION.md`). Nothing here
//! ratifies it; this module exists to price the AIR, not to endorse the hash.

/// The BLAKE3 IV (identical to SHA-256's initial state). `IV[0..4]` seeds
/// `v[8..12]` of the compression working state.
pub const BLAKE3_IV: [u32; 8] = [
    0x6A09E667, 0xBB67AE85, 0x3C6EF372, 0xA54FF53A, 0x510E527F, 0x9B05688C, 0x1F83D9AB, 0x5BE0CD19,
];

/// The BLAKE3 message-schedule permutation, applied between rounds
/// (`m'[i] = m[MSG_PERMUTATION[i]]`).
pub const BLAKE3_MSG_PERMUTATION: [usize; 16] =
    [2, 6, 3, 10, 7, 0, 4, 13, 1, 11, 12, 5, 9, 14, 15, 8];

/// Rounds of the internal variant. 6, per #903's design; standard BLAKE3 is 7.
pub const BLAKE3_ROUNDS: usize = 6;

/// The BLAKE3 quarter-round G (spec §2.1).
#[inline]
fn blake3_g(v: &mut [u32; 16], a: usize, b: usize, c: usize, d: usize, mx: u32, my: u32) {
    v[a] = v[a].wrapping_add(v[b]).wrapping_add(mx);
    v[d] = (v[d] ^ v[a]).rotate_right(16);
    v[c] = v[c].wrapping_add(v[d]);
    v[b] = (v[b] ^ v[c]).rotate_right(12);
    v[a] = v[a].wrapping_add(v[b]).wrapping_add(my);
    v[d] = (v[d] ^ v[a]).rotate_right(8);
    v[c] = v[c].wrapping_add(v[d]);
    v[b] = (v[b] ^ v[c]).rotate_right(7);
}

/// The BLAKE3 compression function `f` at 6 rounds (spec §2.2, oracle §2.4).
///
/// State init: `v[0..8] = h`, `v[8..12] = IV[0..4]`, `v[12] = t as u32`,
/// `v[13] = (t >> 32) as u32`, `v[14] = block_len`, `v[15] = flags`. Six rounds
/// of 8 G-calls (4 columns then 4 diagonals), permuting the message schedule
/// between rounds (`r < rounds - 1`, i.e. 5 permutes — the trailing permute is
/// never consumed). Feed-forward: `out[i] = v[i] ^ v[i+8]`,
/// `out[i+8] = v[i+8] ^ h[i]`. The truncated chaining value is `out[0..8]`.
pub fn blake3_compress_6round(
    h: &[u32; 8],
    m: &[u32; 16],
    t: u64,
    block_len: u32,
    flags: u32,
) -> [u32; 16] {
    let mut v: [u32; 16] = [
        h[0],
        h[1],
        h[2],
        h[3],
        h[4],
        h[5],
        h[6],
        h[7],
        BLAKE3_IV[0],
        BLAKE3_IV[1],
        BLAKE3_IV[2],
        BLAKE3_IV[3],
        t as u32,
        (t >> 32) as u32,
        block_len,
        flags,
    ];

    let mut m = *m;
    for r in 0..BLAKE3_ROUNDS {
        // Mix the columns.
        blake3_g(&mut v, 0, 4, 8, 12, m[0], m[1]);
        blake3_g(&mut v, 1, 5, 9, 13, m[2], m[3]);
        blake3_g(&mut v, 2, 6, 10, 14, m[4], m[5]);
        blake3_g(&mut v, 3, 7, 11, 15, m[6], m[7]);
        // Mix the diagonals.
        blake3_g(&mut v, 0, 5, 10, 15, m[8], m[9]);
        blake3_g(&mut v, 1, 6, 11, 12, m[10], m[11]);
        blake3_g(&mut v, 2, 7, 8, 13, m[12], m[13]);
        blake3_g(&mut v, 3, 4, 9, 14, m[14], m[15]);
        // Permute between rounds; the permute after the last round is never
        // consumed (oracle: `r < rounds - 1`).
        if r < BLAKE3_ROUNDS - 1 {
            let prev = m;
            for (i, &p) in BLAKE3_MSG_PERMUTATION.iter().enumerate() {
                m[i] = prev[p];
            }
        }
    }

    let mut out = [0u32; 16];
    for i in 0..8 {
        out[i] = v[i] ^ v[i + 8];
        out[i + 8] = v[i + 8] ^ h[i];
    }
    out
}

/// One canonical 6-round vector: the oracle's inputs and its 16-word output.
#[derive(Debug, Clone, Copy)]
pub struct Vector {
    pub h: [u32; 8],
    pub m: [u32; 16],
    pub t: u64,
    pub block_len: u32,
    pub flags: u32,
    pub out: [u32; 16],
}

/// The 10 canonical 6-round vectors, transcribed verbatim from #903's
/// `thoughts/blake3/blake3-oracle/canonical_6round_vectors.json` (head
/// `89aeeb8c`). Seeds 0..9 of the oracle's generator; `t` is full-width, which
/// matters — the counter-split order is load-bearing and was behaviourally
/// verified against the official crate.
pub const CANONICAL_VECTORS: [Vector; 10] = [
    Vector {
        h: [
            0xD82C07CD, 0x6BAA9455, 0x82E2E662, 0x7A024204, 0xE87A1613, 0x81332876, 0x48268673,
            0xC17C6279,
        ],
        m: [
            0xE6F4590B, 0x4F65D4D9, 0xBAD640FB, 0xAF19922A, 0x19C78DF4, 0x6F25E2A2, 0xE9BB17BC,
            0x7A1D5006, 0x42AF9FC3, 0x03983CA8, 0xDE1B372A, 0xDED733E8, 0x9148624F, 0xF7B0B7D2,
            0x72AE2244, 0xEECE328B,
        ],
        t: 0xB4E1357D4A84EB03,
        block_len: 42,
        flags: 52,
        out: [
            0xCED9D1FF, 0xC248EEAB, 0xBD109B7F, 0x911B48F6, 0x923D62C0, 0xD804903F, 0x5974223E,
            0xAA4F0C80, 0xAD61007F, 0xB50B8DDB, 0xE7372BE1, 0x33D3D6C3, 0x42AA284B, 0xC5A25F28,
            0x79AC8370, 0xB75F3915,
        ],
    },
    Vector {
        h: [
            0xC386BBC4, 0x414C343C, 0x7311D8A3, 0xA6CECC1B, 0xC9E9C616, 0x18072E8C, 0xD5F4B3B2,
            0x7204E52D,
        ],
        m: [
            0xF1FD42A2, 0xE6C3F339, 0x07D4BEDC, 0x8A9A021E, 0x3BAB6C39, 0x05805975, 0xA46D6753,
            0xDC2574BD, 0xAB99254A, 0x4DA98F1D, 0xE1EA24C4, 0x815A47C5, 0x08D6AF57, 0xCC22AF58,
            0x2C4A3698, 0x5FEC898F,
        ],
        t: 0xC74803E31BA16215,
        block_len: 50,
        flags: 94,
        out: [
            0xF2A972E9, 0x81FDB8EC, 0x40C50EBC, 0x4BA1CAF9, 0x9EE9E930, 0x6B1A16B2, 0xE9156F47,
            0xA89FB436, 0xA2F616B3, 0x12874C12, 0x30768035, 0xE01A17D9, 0xBEE5C17C, 0xD61C0BE0,
            0x3041FF46, 0xDFB91125,
        ],
    },
    Vector {
        h: [
            0x0E7A269F, 0x15BA2BDD, 0xD5E34124, 0x4EE207F8, 0x9B1F282E, 0x9B575BD1, 0xF30B94FA,
            0x0706A045,
        ],
        m: [
            0x6148A86F, 0x8697BBD0, 0x8F7D9B78, 0x3C729578, 0x061B9030, 0x533C9135, 0x829E07B0,
            0xE4C11AB2, 0xCBF87544, 0xC34C769F, 0x5A91C89B, 0xF63F23D0, 0xC1066932, 0x87C56473,
            0x7D718D73, 0xECC1CB63,
        ],
        t: 0x7604E4B4E73695C3,
        block_len: 58,
        flags: 124,
        out: [
            0x5AA6B114, 0xC9D6740C, 0x8738CAF4, 0xAC5F4B72, 0x9FC6B9DE, 0x3F2EFB8F, 0x8CB7A912,
            0xF497A285, 0x3D062266, 0x7F22380C, 0xAFD468FA, 0x122CBA80, 0x446B156D, 0xB239D8C2,
            0xC3EAB2CF, 0x775F2F92,
        ],
    },
    Vector {
        h: [
            0x8B529B4A, 0x9A9A80FD, 0xD6645FA9, 0x3BFD1D33, 0x79F248B0, 0x268ECC45, 0xA2863A7F,
            0x85EF3430,
        ],
        m: [
            0xBDC2AE99, 0x10645D51, 0x97524D6A, 0xDD933160, 0xE0F9E038, 0xEBCD1F5E, 0xEF829C88,
            0xE0FD67DD, 0x18F2C41C, 0x22CEDAFB, 0x378C74DC, 0x4D100D8F, 0x95C76AB4, 0x95918694,
            0xE779C470, 0xEDCF6109,
        ],
        t: 0x92D3043AFCF249F3,
        block_len: 36,
        flags: 31,
        out: [
            0xEED92FAB, 0x138D9358, 0x915BFE3C, 0x13718B01, 0xB506E277, 0xBE4007CD, 0x35847E06,
            0xCE1C6896, 0x52FA01B5, 0x4AA26AF8, 0xB1078A61, 0x2C517AED, 0xA08867A0, 0xEA6ECFEA,
            0x6D33D3B0, 0xDC293166,
        ],
    },
    Vector {
        h: [
            0x3C6DA5D7, 0x656412A9, 0x27AC435A, 0x11072231, 0xEAFF1A09, 0xC3E1B258, 0x8963DC6E,
            0x1B2ED40E,
        ],
        m: [
            0xED6F0B09, 0xCE80C4B0, 0xCCEA2645, 0x3184FF27, 0x4F5253A0, 0xE14B0190, 0x9B191BF4,
            0xABF4A07C, 0x81862FC9, 0x2D83A823, 0x793D0E45, 0x4CDCE7A6, 0xE8ABB93F, 0xE1DF8AF9,
            0x8224B122, 0x69F85E31,
        ],
        t: 0x49C7B59B995253FD,
        block_len: 57,
        flags: 41,
        out: [
            0xCA00BDA3, 0x84239A3A, 0xE7C88E6D, 0x33A8A3D6, 0x09DCD1CE, 0xA1B10212, 0xF48E1156,
            0x8F039915, 0x8A055EAA, 0xFF5B11D5, 0xB725085B, 0x2E1AB267, 0x6AE7323D, 0xB2FF6FA8,
            0x7102C8A1, 0x7561EB37,
        ],
    },
    Vector {
        h: [
            0x9F767C45, 0xBDE5C099, 0xF17FD374, 0xA6233255, 0xE6A16A3B, 0x1CFB10F6, 0x3F1F65A8,
            0x8B33E968,
        ],
        m: [
            0x92EDCF45, 0x377B9AA2, 0x478C281D, 0xC4069545, 0xCC11D357, 0x9E115E4B, 0x206F5C66,
            0xDF1461AA, 0xFB7FF337, 0xDF561D80, 0x4A0FE75D, 0xF6236BF2, 0x346C6E2B, 0xB0CDE917,
            0xE4CC4132, 0x4C7D6DF0,
        ],
        t: 0x6A3753915C76F18A,
        block_len: 18,
        flags: 67,
        out: [
            0x14A9F66F, 0x101BDFE8, 0x9B0A50DD, 0xEE4BB45B, 0x7A914502, 0x77B3486B, 0x59BFC114,
            0xA1AD2AFD, 0xC194DDE6, 0x894EC54D, 0xAD36C805, 0x9018F3F5, 0x165AF5D8, 0x3E85B598,
            0x78E76653, 0xBB7A485D,
        ],
    },
    Vector {
        h: [
            0xD26B9496, 0x42F9A039, 0x001D9A88, 0x5F877031, 0xC527E279, 0x45CF8AA4, 0xCD4A5557,
            0xAE9AF169,
        ],
        m: [
            0xAF895F5B, 0xD822E2F9, 0x17D7AB26, 0xCCDF540B, 0xCE06294D, 0x4A8B0188, 0xF38D2E64,
            0x5C41D5C5, 0xE8D5B9E3, 0x5C832A51, 0x9A0C1B76, 0x4DE8344E, 0x96D2F9E0, 0x8677A5F2,
            0xA9A967C1, 0x323BBEAF,
        ],
        t: 0x390567C27BD6AA42,
        block_len: 26,
        flags: 3,
        out: [
            0x32A6FF70, 0xC30560BC, 0xD1C777C8, 0xF1871821, 0x7207AB54, 0x9F5B83C7, 0xB6561C5D,
            0x991E738F, 0xB38B62B9, 0x0EF6D156, 0x994BECB1, 0x09A85D0E, 0x32221741, 0xADA3CC5F,
            0x5B654ED6, 0x2A7A62B2,
        ],
    },
    Vector {
        h: [
            0x269E0D37, 0xA6A3A450, 0x892F902B, 0x81E74EF5, 0x099950D8, 0x6F03675A, 0x11E20B8F,
            0x6CAD4A26,
        ],
        m: [
            0xF29D0DA9, 0x658CDA14, 0xF9EBDACC, 0xDBC496CB, 0x4A23D596, 0x2E44158B, 0xA38FD547,
            0x5F557203, 0x34B9B5DF, 0x506BF2EF, 0x7403E430, 0x4CBD87AD, 0xCB5C7427, 0x3E7D1BFB,
            0x930D6EAF, 0x86734721,
        ],
        t: 0x12BD4ACEFAECBD38,
        block_len: 53,
        flags: 42,
        out: [
            0xA632AD45, 0x12CE41F4, 0xD21B2CBD, 0x76795C62, 0x6BEC36C1, 0xDAFAFCDE, 0x53CA87B7,
            0x92E8465B, 0x7B424F5D, 0xE1E6AD7F, 0x753BA387, 0xCCC50824, 0x69AEDF6D, 0xBBBBF253,
            0x78D04883, 0xF3F33689,
        ],
    },
    Vector {
        h: [
            0x3A096533, 0xF658F7A7, 0x205738D1, 0xB46EE1DA, 0x15CEB3A1, 0x359B1548, 0xA4517D6C,
            0x7589CA4A,
        ],
        m: [
            0x74007CB4, 0xD49D0AC1, 0x16EDC5D4, 0x685CA8AF, 0x4223AA56, 0x10269470, 0x60908405,
            0xA92D04A3, 0x56A3E957, 0xB0F91306, 0xE6C08269, 0xF2306D4A, 0x31A06A7C, 0x9436D6F6,
            0xE18692E2, 0xE0C99F3E,
        ],
        t: 0x329911DA9FBD8735,
        block_len: 19,
        flags: 91,
        out: [
            0x913B2AE1, 0xC7F73082, 0x45E1C023, 0x6F1F3F82, 0x20AEE6F5, 0xDAF21D94, 0xF2C1E4AF,
            0xD4F7D4AC, 0x44A45F87, 0xF4C40CE5, 0x613E9B94, 0x08CE53DE, 0x4FF07AA4, 0x456BF2E2,
            0x2066EA7F, 0x3C5A654B,
        ],
    },
    Vector {
        h: [
            0x5F915EF0, 0x237751AA, 0x01A5BA50, 0x80B65386, 0x14B044D7, 0x61076DC3, 0xB99DE255,
            0x283B73A6,
        ],
        m: [
            0x3CEE5E2C, 0x1C670EA9, 0x972651DA, 0x4A8AA593, 0xAC9ABB0C, 0x35BB5C11, 0x47FBB3B4,
            0xCF3C17E5, 0xE2EB17C8, 0xE11E99FB, 0x7DE0D208, 0x0602FE0C, 0x98CAE043, 0x9425B3E2,
            0x33FB4B4F, 0x15607DF9,
        ],
        t: 0xEAEB999B8A2E547E,
        block_len: 64,
        flags: 21,
        out: [
            0xF5EE9114, 0x856CABB8, 0x29BE2CF1, 0x603BE91C, 0x94A7DD0E, 0x28FC3E27, 0xB64E2CC8,
            0x2D2C67FF, 0x69FAC1BA, 0x0C949090, 0xD68DE435, 0xCE91A527, 0xE80C1815, 0x6D44EFE6,
            0x87C7B175, 0xD18A8B94,
        ],
    },
];

#[cfg(test)]
mod tests {
    use super::*;

    /// The conventions a wrong port could get wrong, as data.
    ///
    /// [`CANONICAL_VECTORS`] is supposed to pin every one of these. Naming them
    /// in a struct is what lets the negative control break exactly one at a time.
    #[derive(Clone, Copy)]
    struct Conventions {
        /// The four rotation amounts of `G`, in application order.
        rot: [u32; 4],
        /// The message-schedule permutation applied between rounds.
        perm: [usize; 16],
        rounds: usize,
    }

    const CANONICAL: Conventions = Conventions {
        rot: [16, 12, 8, 7],
        perm: BLAKE3_MSG_PERMUTATION,
        rounds: BLAKE3_ROUNDS,
    };

    /// A deliberately *parameterised* compression, used only to build negative
    /// controls: the same dataflow with [`Conventions`] as an input.
    ///
    /// It is NOT what [`blake3_compress_6round`] calls. Keeping the two apart
    /// costs a duplicated loop and buys the thing rule 7 is about: the control
    /// tests below compare this function's output against [`CANONICAL_VECTORS`]
    /// — a constant that came from outside this file — so they stay meaningful
    /// no matter how the real function is later refactored.
    fn compress_variant(v: &Vector, c: Conventions) -> [u32; 16] {
        let g = |s: &mut [u32; 16], a: usize, b: usize, cc: usize, d: usize, mx: u32, my: u32| {
            s[a] = s[a].wrapping_add(s[b]).wrapping_add(mx);
            s[d] = (s[d] ^ s[a]).rotate_right(c.rot[0]);
            s[cc] = s[cc].wrapping_add(s[d]);
            s[b] = (s[b] ^ s[cc]).rotate_right(c.rot[1]);
            s[a] = s[a].wrapping_add(s[b]).wrapping_add(my);
            s[d] = (s[d] ^ s[a]).rotate_right(c.rot[2]);
            s[cc] = s[cc].wrapping_add(s[d]);
            s[b] = (s[b] ^ s[cc]).rotate_right(c.rot[3]);
        };
        let h = v.h;
        let mut s: [u32; 16] = [
            h[0],
            h[1],
            h[2],
            h[3],
            h[4],
            h[5],
            h[6],
            h[7],
            BLAKE3_IV[0],
            BLAKE3_IV[1],
            BLAKE3_IV[2],
            BLAKE3_IV[3],
            v.t as u32,
            (v.t >> 32) as u32,
            v.block_len,
            v.flags,
        ];
        let mut m = v.m;
        for r in 0..c.rounds {
            g(&mut s, 0, 4, 8, 12, m[0], m[1]);
            g(&mut s, 1, 5, 9, 13, m[2], m[3]);
            g(&mut s, 2, 6, 10, 14, m[4], m[5]);
            g(&mut s, 3, 7, 11, 15, m[6], m[7]);
            g(&mut s, 0, 5, 10, 15, m[8], m[9]);
            g(&mut s, 1, 6, 11, 12, m[10], m[11]);
            g(&mut s, 2, 7, 8, 13, m[12], m[13]);
            g(&mut s, 3, 4, 9, 14, m[14], m[15]);
            if r < c.rounds - 1 {
                let prev = m;
                for (i, &p) in c.perm.iter().enumerate() {
                    m[i] = prev[p];
                }
            }
        }
        let mut out = [0u32; 16];
        for i in 0..8 {
            out[i] = s[i] ^ s[i + 8];
            out[i + 8] = s[i + 8] ^ h[i];
        }
        out
    }

    /// The port reproduces all ten canonical vectors.
    #[test]
    fn the_compression_matches_the_canonical_six_round_vectors() {
        for (i, v) in CANONICAL_VECTORS.iter().enumerate() {
            assert_eq!(
                blake3_compress_6round(&v.h, &v.m, v.t, v.block_len, v.flags),
                v.out,
                "canonical 6-round vector {i}"
            );
        }
    }

    /// The parameterised control, at canonical parameters, IS the port — so a
    /// negative control below differs from the real thing in exactly the one
    /// convention it names, and nothing else.
    #[test]
    fn the_variant_at_canonical_parameters_is_the_port() {
        for v in CANONICAL_VECTORS.iter() {
            assert_eq!(
                compress_variant(v, CANONICAL),
                v.out,
                "the control must reproduce the vectors at canonical parameters"
            );
        }
    }

    /// NEGATIVE CONTROL (rule 9): each convention the vectors are supposed to
    /// pin, broken one at a time, must stop reproducing them.
    ///
    /// Without this, "the vectors pass" would be evidence only that the vectors
    /// are *reachable*, not that they discriminate. Each case names what would
    /// silently be unpinned if it ever started passing.
    #[test]
    fn breaking_one_convention_at_a_time_breaks_the_vectors() {
        // The message permutation transposed (its own inverse composition):
        // same multiset of indices, same round count, different schedule.
        let mut transposed = [0usize; 16];
        for (i, &p) in BLAKE3_MSG_PERMUTATION.iter().enumerate() {
            transposed[p] = i;
        }
        let cases: [(&str, Conventions); 4] = [
            // rotr12 -> rotr13: the one rotation amount that is NOT a byte
            // relabel in the chip, so a wrong value here is the wrong-rotation
            // bug in its most consequential place.
            (
                "rotr12 -> rotr13",
                Conventions {
                    rot: [16, 13, 8, 7],
                    ..CANONICAL
                },
            ),
            // rotr16 and rotr8 swapped: both ARE free byte relabels in the
            // chip, so transposing them costs no columns and no constraints —
            // the cheapest possible way to be wrong.
            (
                "rotr16 <-> rotr8",
                Conventions {
                    rot: [8, 12, 16, 7],
                    ..CANONICAL
                },
            ),
            (
                "message schedule transposed",
                Conventions {
                    perm: transposed,
                    ..CANONICAL
                },
            ),
            (
                "7 rounds (standard BLAKE3)",
                Conventions {
                    rounds: 7,
                    ..CANONICAL
                },
            ),
        ];
        for (what, c) in cases {
            let v = &CANONICAL_VECTORS[0];
            assert_ne!(
                compress_variant(v, c),
                v.out,
                "{what} still reproduces the canonical vector — the vector does not pin it"
            );
        }
    }

    /// The counter split is load-bearing and full-width: `t` reaches the state
    /// as two 32-bit halves in low-then-high order, so swapping them must move
    /// the output. Six of the ten canonical vectors have distinct halves.
    #[test]
    fn the_counter_halves_are_not_interchangeable() {
        let mut checked = 0;
        for v in CANONICAL_VECTORS.iter() {
            let swapped = v.t.rotate_left(32);
            if swapped == v.t {
                continue;
            }
            checked += 1;
            assert_ne!(
                blake3_compress_6round(&v.h, &v.m, swapped, v.block_len, v.flags),
                v.out,
                "swapping the counter halves must change the output"
            );
        }
        assert!(
            checked >= 8,
            "expected most vectors to have distinct halves, got {checked}"
        );
    }
}
