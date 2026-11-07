//! LEAF-mode KATs for the LFM `"LFML"` domain, at 6 and 7 rounds.
//!
//! GENERATED — do not hand-edit. Rendered by
//! `thoughts/shared/lfm-real-hash/leaf-spec/rate4_kat_gen.py` from
//! `gate-oracle/blake3_oracle.py`, a Python BLAKE3 written **before any Rust
//! existed**. These vectors are a specification the implementation is checked
//! against, not a recording of what the implementation happened to do.
//!
//! A leaf row hashes FOUR arbitrary Goldilocks elements AND chains an
//! accumulator, in ONE compression (COMMIT.md §1.2). The accumulator is a digest
//! cell and fills lanes 0–3; each felt occupies two lanes above it as checked
//! `u32` halves, `[lo0, hi0, …, lo3, hi3]`. So the message is
//! `LE32(acc ‖ halves) ‖ "LFML"` — 52 bytes, still one BLAKE3 block, so the
//! crate-KAT anchor survives the widening.

/// One leaf row: the chaining accumulator, four felts, the twelve lanes they
/// become, and the digest at each round count.
pub struct LeafVector {
    pub name: &'static str,
    pub acc: [u32; 4],
    pub felts: [u64; 4],
    pub lanes: [u32; 12],
    /// Digest at 6 rounds (the A6R variant; no library computes it).
    pub digest_6: [u32; 4],
    /// Digest at 7 rounds — `blake3::hash(LE32(lanes) ‖ "LFML")[..16]`.
    pub digest_7: [u32; 4],
}

pub const LEAF_VECTORS: [LeafVector; 6] = [
    LeafVector {
        name: "zeros",
        acc: [0x00000000, 0x00000000, 0x00000000, 0x00000000],
        felts: [0u64, 0u64, 0u64, 0u64],
        lanes: [
            0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
            0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
        ],
        digest_6: [0x9D79DC29, 0xFC6E166E, 0x30387614, 0xF6B51296],
        digest_7: [0xB30DB92A, 0xC648E66E, 0x85368146, 0x30A98B38],
    },
    LeafVector {
        name: "boundary_mix",
        acc: [0x00000000, 0x00000001, 0xFFFFFFFE, 0xFFFFFFFF],
        felts: [0u64, 1u64, 18446744069414584320u64, 4294967296u64],
        lanes: [
            0x00000000, 0x00000001, 0xFFFFFFFE, 0xFFFFFFFF, 0x00000000, 0x00000000, 0x00000001,
            0x00000000, 0x00000000, 0xFFFFFFFF, 0x00000000, 0x00000001,
        ],
        digest_6: [0xA101443C, 0xA70F5A93, 0xAD973E8C, 0x17C8F7BA],
        digest_7: [0x0E214E2C, 0x5D16CE5C, 0xA4DE74CF, 0x9FA39D59],
    },
    LeafVector {
        name: "all_p_minus_1",
        acc: [0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF],
        felts: [
            18446744069414584320u64,
            18446744069414584320u64,
            18446744069414584320u64,
            18446744069414584320u64,
        ],
        lanes: [
            0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0x00000000, 0xFFFFFFFF, 0x00000000,
            0xFFFFFFFF, 0x00000000, 0xFFFFFFFF, 0x00000000, 0xFFFFFFFF,
        ],
        digest_6: [0x5FDFAEF4, 0x0F63DEEE, 0xCC4EC296, 0x675289C3],
        digest_7: [0xAE3CB971, 0x6F0EDEE7, 0x75BBD078, 0xC07D12D6],
    },
    LeafVector {
        name: "ramp",
        acc: [0x01020304, 0x05060708, 0x090A0B0C, 0x0D0E0F10],
        felts: [
            72623859790382856u64,
            1230066625199609624u64,
            2387509390608836392u64,
            3544952156018063160u64,
        ],
        lanes: [
            0x01020304, 0x05060708, 0x090A0B0C, 0x0D0E0F10, 0x05060708, 0x01020304, 0x15161718,
            0x11121314, 0x25262728, 0x21222324, 0x35363738, 0x31323334,
        ],
        digest_6: [0x7E8EC742, 0x478136B7, 0xDC4010C2, 0xA7B85A1F],
        digest_7: [0x6F6562C3, 0x6755528E, 0xBD65A6F0, 0xA9B1551D],
    },
    LeafVector {
        name: "u32_edges",
        acc: [0x80000000, 0x7FFFFFFF, 0x00010000, 0x0000FFFF],
        felts: [4294967295u64, 4294967296u64, 18446744065119617025u64, 1u64],
        lanes: [
            0x80000000, 0x7FFFFFFF, 0x00010000, 0x0000FFFF, 0xFFFFFFFF, 0x00000000, 0x00000000,
            0x00000001, 0x00000001, 0xFFFFFFFE, 0x00000001, 0x00000000,
        ],
        digest_6: [0x0E17BDDF, 0x1E0CA3B6, 0x7F8B414F, 0xDDF551B2],
        digest_7: [0x77E4CDFD, 0x92CC8E05, 0x1BBC4BD0, 0x64B4D8D2],
    },
    LeafVector {
        name: "acc_ignored_control",
        acc: [0x11121314, 0x15161718, 0x191A1B1C, 0x1D1E1F20],
        felts: [0u64, 0u64, 0u64, 0u64],
        lanes: [
            0x11121314, 0x15161718, 0x191A1B1C, 0x1D1E1F20, 0x00000000, 0x00000000, 0x00000000,
            0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
        ],
        digest_6: [0xE4AE2501, 0x1FCF9DAB, 0x85643F4E, 0xE24B3793],
        digest_7: [0xB8094093, 0xA7EBC1A4, 0xB7955183, 0x0BA8929B],
    },
];

/// A boundary felt and the halves it must decompose into.
pub struct BoundaryFelt {
    pub name: &'static str,
    pub felt: u64,
    pub lo: u32,
    pub hi: u32,
}

/// The six canonical boundary cases, `p − 1` included — the tight one.
pub const BOUNDARY_FELTS: [BoundaryFelt; 6] = [
    BoundaryFelt {
        name: "zero",
        felt: 0u64,
        lo: 0x00000000,
        hi: 0x00000000,
    },
    BoundaryFelt {
        name: "one",
        felt: 1u64,
        lo: 0x00000001,
        hi: 0x00000000,
    },
    BoundaryFelt {
        name: "u32_max",
        felt: 4294967295u64,
        lo: 0xFFFFFFFF,
        hi: 0x00000000,
    },
    BoundaryFelt {
        name: "two_pow_32",
        felt: 4294967296u64,
        lo: 0x00000000,
        hi: 0x00000001,
    },
    BoundaryFelt {
        name: "p_minus_2_32",
        felt: 18446744065119617025u64,
        lo: 0x00000001,
        hi: 0xFFFFFFFE,
    },
    BoundaryFelt {
        name: "p_minus_1",
        felt: 18446744069414584320u64,
        lo: 0x00000000,
        hi: 0xFFFFFFFF,
    },
];

/// Values the leaf mode must REJECT rather than reduce. Each has `hi` maximal
/// and `lo >= 1`, so each aliases a canonical felt — which is exactly the
/// collision the canonicity block exists to prevent.
pub struct NonCanonical {
    pub name: &'static str,
    pub value: u64,
    pub lo: u32,
    pub hi: u32,
}

pub const NON_CANONICAL: [NonCanonical; 3] = [
    NonCanonical {
        name: "p",
        value: 18446744069414584321u64,
        lo: 0x00000001,
        hi: 0xFFFFFFFF,
    },
    NonCanonical {
        name: "p_plus_1",
        value: 18446744069414584322u64,
        lo: 0x00000002,
        hi: 0xFFFFFFFF,
    },
    NonCanonical {
        name: "two_pow_64_minus_1",
        value: 18446744073709551615u64,
        lo: 0xFFFFFFFF,
        hi: 0xFFFFFFFF,
    },
];

/// The eight-felt `FriToyV0` leaf: ONE `LFML` chain, two rows, no fold.
pub struct FriLeafVector {
    pub felts: [u64; 8],
    pub digest_6: [u32; 4],
    pub digest_7: [u32; 4],
    /// Compressions the whole leaf costs — 3 before the accumulator moved into
    /// the message, 2 after (COMMIT.md §1.4.1: the RATE, measured).
    pub compresses: usize,
}

pub const FRI_LEAF: FriLeafVector = FriLeafVector {
    felts: [
        18446744069414584320u64,
        0u64,
        1u64,
        4294967296u64,
        12345678901234567u64,
        4294967295u64,
        18446744065119617025u64,
        999u64,
    ],
    digest_6: [0x8578A6BC, 0x9160F074, 0x3F4C82B9, 0x98C5C775],
    digest_7: [0x9C36DE23, 0xCD397230, 0x2013BF3D, 0xD72A0346],
    compresses: 2,
};
