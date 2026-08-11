//! Socket KATs for the LFM 2-to-1 BLAKE3 compress, at 6 and 7 rounds.
//!
//! GENERATED — do not hand-edit. The union of two independently produced
//! vector tables: `thoughts/blake3/socket-kats/socket_kats.json` (Phase 1,
//! upstream BLAKE3's C at word level + its whole tree hasher at byte level)
//! and the gate-oracle's `socket_kats.json` (a separately written Python
//! oracle). The two share 5 of the 15 input pairs and agree on every one of
//! them at both round counts, which is what makes this table two sources
//! rather than one transcribed twice.
//!
//! Framing (`SOCKET.md` §2.2): h = BLAKE3_IV, m[0..4] = a, m[4..8] = b,
//! m[8] = "LFMC", m[9..16] = 0, t = 0, block_len = 36, flags = 0x0B,
//! digest = out[0..4].

/// One socket vector: the two input cells and the digest at each round count.
pub struct SocketVector {
    pub name: &'static str,
    pub a: [u32; 4],
    pub b: [u32; 4],
    /// Digest at 6 rounds (the A6R variant; no library computes it).
    pub digest_6: [u32; 4],
    /// Digest at 7 rounds — `blake3::hash(a ‖ b ‖ "LFMC")[..16]`.
    pub digest_7: [u32; 4],
}

pub const SOCKET_VECTORS: [SocketVector; 15] = [
    SocketVector {
        name: "a_one/unit_a", // socket-kats+gate-oracle
        a: [0x00000001, 0x00000000, 0x00000000, 0x00000000],
        b: [0x00000000, 0x00000000, 0x00000000, 0x00000000],
        digest_6: [0xD41793E6, 0x43B503F8, 0x701C3D9A, 0x21761C9D],
        digest_7: [0xB9046BC7, 0x75AF9CBE, 0xABD6AA9C, 0x675F0135],
    },
    SocketVector {
        name: "all_ones/all_ones", // socket-kats+gate-oracle
        a: [0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF],
        b: [0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF],
        digest_6: [0x1CB39655, 0x466D7CA9, 0xA8E4463E, 0x4D33195F],
        digest_7: [0x1E8635C0, 0x82CC223F, 0xC6F238CA, 0x57F9C01F],
    },
    SocketVector {
        name: "b_one/unit_b", // socket-kats+gate-oracle
        a: [0x00000000, 0x00000000, 0x00000000, 0x00000000],
        b: [0x00000001, 0x00000000, 0x00000000, 0x00000000],
        digest_6: [0x9204D33A, 0xC3CE7C22, 0x023D1838, 0x39247C70],
        digest_7: [0x86D366C5, 0xE620A872, 0xF8340F7B, 0xD08847C1],
    },
    SocketVector {
        name: "boundary", // gate-oracle
        a: [0x00000000, 0x00000001, 0xFFFFFFFE, 0xFFFFFFFF],
        b: [0x80000000, 0x7FFFFFFF, 0x00010000, 0x0000FFFF],
        digest_6: [0x3CC67FDE, 0xFF4C844D, 0xDC443911, 0xDB69BC03],
        digest_7: [0xEBE12135, 0x3E108A3A, 0xEBDB7879, 0x870F5210],
    },
    SocketVector {
        name: "formula_0", // socket-kats
        a: [0x9E3779B9, 0x3C6EF372, 0xDAA66D2B, 0x78DDE6E4],
        b: [0x8FF34781, 0x2E2AC13A, 0xCC623AF3, 0x6A99B4AC],
        digest_6: [0x2B537C88, 0xFA6F602A, 0x5DDE2DB6, 0x7B394E5C],
        digest_7: [0x50972B73, 0x4AF4BF2C, 0x934921FD, 0x3A5C35C6],
    },
    SocketVector {
        name: "formula_1", // socket-kats
        a: [0x81AF1549, 0x1FE68F02, 0xBE1E08BB, 0x5C558274],
        b: [0x736AE311, 0x11A25CCA, 0xAFD9D683, 0x4E11503C],
        digest_6: [0xF00DB14B, 0x49B031E4, 0x1DDB8781, 0x37502416],
        digest_7: [0x07E72396, 0x91999572, 0x81B79946, 0x48615641],
    },
    SocketVector {
        name: "formula_1", // gate-oracle
        a: [0x01020304, 0x05060708, 0x090A0B0C, 0x0D0E0F10],
        b: [0x11121314, 0x15161718, 0x191A1B1C, 0x1D1E1F20],
        digest_6: [0xBA2A1897, 0x7545C999, 0xB7269CDA, 0x8A29378F],
        digest_7: [0x0E4191BB, 0x4D281F4F, 0x61E3612C, 0x49DE2543],
    },
    SocketVector {
        name: "formula_2", // socket-kats
        a: [0x6526B0D9, 0x035E2A92, 0xA195A44B, 0x3FCD1E04],
        b: [0x56E27EA1, 0xF519F85A, 0x93517213, 0x3188EBCC],
        digest_6: [0x9BAB44A9, 0x9A2594F1, 0xDDD9BD99, 0xF1781997],
        digest_7: [0xCBAA622A, 0x0C3FEFE7, 0x31E864BB, 0x63D5371C],
    },
    SocketVector {
        name: "formula_2", // gate-oracle
        a: [0xDEADBEEF, 0xCAFEBABE, 0x8BADF00D, 0xFEEDFACE],
        b: [0x0BADC0DE, 0xD15EA5E5, 0xC0FFEE00, 0xBAAAAAAD],
        digest_6: [0x2979A598, 0x77AF5CDE, 0x57855D1B, 0x30A0B8B7],
        digest_7: [0xEA710B4F, 0x620D78A5, 0xD7168741, 0x451B44C6],
    },
    SocketVector {
        name: "formula_3", // socket-kats
        a: [0x489E4C69, 0xE6D5C622, 0x850D3FDB, 0x2344B994],
        b: [0x3A5A1A31, 0xD89193EA, 0x76C90DA3, 0x1500875C],
        digest_6: [0xC3D20C43, 0x692332E6, 0x79B8D6E1, 0xBBDFF098],
        digest_7: [0xAD449F12, 0x2BA1E6B4, 0x23FF2A55, 0xFE81E452],
    },
    SocketVector {
        name: "formula_3", // gate-oracle
        a: [0x7F800001, 0x00000002, 0x80000000, 0x7FFFFFFF],
        b: [0x00FF00FF, 0xFF00FF00, 0x0F0F0F0F, 0xF0F0F0F0],
        digest_6: [0x0E464B87, 0xFA7E96AE, 0x426B0BFA, 0x7C7A0882],
        digest_7: [0x094C5B0E, 0x19DE2850, 0x7185A7DC, 0x24D73F47],
    },
    SocketVector {
        name: "formula_4", // socket-kats
        a: [0x2C15E7F9, 0xCA4D61B2, 0x6884DB6B, 0x06BC5524],
        b: [0x1DD1B5C1, 0xBC092F7A, 0x5A40A933, 0xF87822EC],
        digest_6: [0xEA4581F0, 0xA5EA3CBC, 0x779EDFCD, 0xC467D11C],
        digest_7: [0x14DDC004, 0xE63073AA, 0x08A8F883, 0xE429C3AA],
    },
    SocketVector {
        name: "max_min", // gate-oracle
        a: [0xFFFFFFFF, 0x00000000, 0xFFFFFFFF, 0x00000000],
        b: [0x00000000, 0xFFFFFFFF, 0x00000000, 0xFFFFFFFF],
        digest_6: [0x43594335, 0xD779C7E8, 0x40424E19, 0x9D340534],
        digest_7: [0xD0722F85, 0x01149B0B, 0xBE0FBEDD, 0x539BE2E5],
    },
    SocketVector {
        name: "nibble_ramp/nibble_ramp", // socket-kats+gate-oracle
        a: [0x00000000, 0x11111111, 0x22222222, 0x33333333],
        b: [0x44444444, 0x55555555, 0x66666666, 0x77777777],
        digest_6: [0x2EF9ED44, 0x4B4AB3F5, 0x6BE64DC6, 0xDABEF7B1],
        digest_7: [0x1AAA3EC0, 0x66DD5B29, 0xE9A45630, 0x61D274FF],
    },
    SocketVector {
        name: "zeros/zeros", // socket-kats+gate-oracle
        a: [0x00000000, 0x00000000, 0x00000000, 0x00000000],
        b: [0x00000000, 0x00000000, 0x00000000, 0x00000000],
        digest_6: [0xA77AF713, 0x8ECE88C9, 0x1918D4BB, 0xF67E206E],
        digest_7: [0x94A80248, 0x30216AFC, 0x8C094E7F, 0xE7A6CC2D],
    },
];
