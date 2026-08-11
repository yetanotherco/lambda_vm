//! LEAF-mode KATs for the LFM `"LFML"` domain, at 6 and 7 rounds.
//!
//! GENERATED — do not hand-edit. Rendered from
//! `thoughts/shared/lfm-real-hash/leaf-spec/leaf_kats.json`, which the oracle
//! produced from a Python reference written **before any Rust existed**. These
//! vectors are a specification the implementation is checked against, not a
//! recording of what the implementation happened to do.
//!
//! A leaf row hashes FOUR arbitrary Goldilocks elements. Each felt occupies two
//! lanes as checked `u32` halves, `[lo0, hi0, …, lo3, hi3]`, so the message
//! layout is byte-identical to a digest-mode compress and the crate-KAT anchor
//! survives untouched.

/// One leaf row: four felts, the eight lanes they become, and the digest at
/// each round count.
pub struct LeafVector {
    pub name: &'static str,
    pub felts: [u64; 4],
    pub lanes: [u32; 8],
    /// Digest at 6 rounds (the A6R variant; no library computes it).
    pub digest_6: [u32; 4],
    /// Digest at 7 rounds — `blake3::hash(LE32(lanes) ‖ "LFML")[..16]`.
    pub digest_7: [u32; 4],
}

pub const LEAF_VECTORS: [LeafVector; 5] = [
    LeafVector {
        name: "zeros",
        felts: [0u64, 0u64, 0u64, 0u64],
        lanes: [
            0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
            0x00000000,
        ],
        digest_6: [0x987496E2, 0x674930D6, 0xD6F9F709, 0xBDFC162E],
        digest_7: [0x3CA2C373, 0x79140765, 0x3E706CB0, 0xE4A11D3A],
    },
    LeafVector {
        name: "boundary_mix",
        felts: [0u64, 1u64, 18446744069414584320u64, 4294967296u64],
        lanes: [
            0x00000000, 0x00000000, 0x00000001, 0x00000000, 0x00000000, 0xFFFFFFFF, 0x00000000,
            0x00000001,
        ],
        digest_6: [0x01A070C2, 0x7758BF44, 0xCED65D68, 0x54FF7227],
        digest_7: [0x43FA6E44, 0xEB0A55F1, 0xAB80535C, 0xB013D578],
    },
    LeafVector {
        name: "all_p_minus_1",
        felts: [
            18446744069414584320u64,
            18446744069414584320u64,
            18446744069414584320u64,
            18446744069414584320u64,
        ],
        lanes: [
            0x00000000, 0xFFFFFFFF, 0x00000000, 0xFFFFFFFF, 0x00000000, 0xFFFFFFFF, 0x00000000,
            0xFFFFFFFF,
        ],
        digest_6: [0x96B22DF5, 0x8F8FFB10, 0x9F7A0569, 0x8A86F904],
        digest_7: [0x16CAF28B, 0x9434478C, 0xC9C723D8, 0x734E72FD],
    },
    LeafVector {
        name: "ramp",
        felts: [
            72623859790382856u64,
            1230066625199609624u64,
            2387509390608836392u64,
            3544952156018063160u64,
        ],
        lanes: [
            0x05060708, 0x01020304, 0x15161718, 0x11121314, 0x25262728, 0x21222324, 0x35363738,
            0x31323334,
        ],
        digest_6: [0x72EB82EF, 0xC66B9255, 0x270356DE, 0xA5A6F3F3],
        digest_7: [0x7588177A, 0x779592F1, 0x96EA4AC5, 0x378E2D2A],
    },
    LeafVector {
        name: "u32_edges",
        felts: [4294967295u64, 4294967296u64, 18446744065119617025u64, 1u64],
        lanes: [
            0xFFFFFFFF, 0x00000000, 0x00000000, 0x00000001, 0x00000001, 0xFFFFFFFE, 0x00000001,
            0x00000000,
        ],
        digest_6: [0x78F2D23E, 0x5E3949A0, 0x3CC550CA, 0xF3A35DEF],
        digest_7: [0x15587203, 0x427A6C0C, 0x99ABD637, 0xD198DFE5],
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

/// The eight-felt `FriToyV0` leaf: two `LFML` rows and one `LFMC` parent.
pub struct FriLeafVector {
    pub felts: [u64; 8],
    pub digest_6: [u32; 4],
    pub digest_7: [u32; 4],
    /// Compressions the whole leaf costs — the ratified 1 → 3.
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
    digest_6: [0xBF4978E9, 0x6E7668FE, 0xCB785244, 0x587400B8],
    digest_7: [0x625237B7, 0x806A7F80, 0xB7D0ABBE, 0x32C418E0],
    compresses: 3,
};
