//! Socket KATs for the LFM 2-to-1 BLAKE3 compress, at 6 and 7 rounds.
//!
//! GENERATED — do not hand-edit. The INPUT pairs are the union of two
//! independently produced vector tables:
//! `thoughts/blake3/socket-kats/socket_kats.json` (Phase 1, upstream BLAKE3's C
//! at word level + its whole tree hasher at byte level) and the gate-oracle's
//! `socket_kats.json` (a separately written Python oracle). The two share 5 of
//! the 15 input pairs and agreed on every one of them at both round counts,
//! which is what makes this table two sources rather than one transcribed
//! twice.
//!
//! ⚠ **The DIGESTS were re-pinned when the socket widened to twelve lanes**, by
//! `leaf-spec/rate4_kat_gen.py` out of that same gate-oracle BLAKE3. All 30
//! moved and no input did: `block_len` is `v[14]` and cannot be made
//! mode-dependent, so the Merkle domain re-blesses alongside the leaf domain
//! that needed the width (COMMIT.md §1.4.4 H9). The upstream-C leg of the
//! provenance is therefore historical for the digests and current for the
//! inputs. What still anchors the new values outside this tree is that a
//! 52-byte message is one BLAKE3 block, so at 7 rounds each vector remains a
//! plain `blake3::hash` call.
//!
//! Framing (`SOCKET.md` §2.2 at the COMMIT.md §1.2 width): h = BLAKE3_IV,
//! m[0..4] = a, m[4..8] = b, m[8..12] = 0 — the third input cell, which the
//! unread-`IN` pins force to zero — m[12] = "LFMC", m[13..16] = 0, t = 0,
//! block_len = 52, flags = 0x0B, digest = out[0..4].

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
        digest_6: [0x34064AA0, 0xD7155685, 0x37B1522B, 0x17147454],
        digest_7: [0xFD482C6D, 0xF3D43A7D, 0xECE55DC3, 0xA2594A53],
    },
    SocketVector {
        name: "all_ones/all_ones", // socket-kats+gate-oracle
        a: [0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF],
        b: [0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF],
        digest_6: [0xCCEF3565, 0x52BAE2BB, 0x4C0E5777, 0x2896C5CB],
        digest_7: [0x6C73358D, 0x7AC15CE6, 0xDB7BFA9C, 0x65CD3364],
    },
    SocketVector {
        name: "b_one/unit_b", // socket-kats+gate-oracle
        a: [0x00000000, 0x00000000, 0x00000000, 0x00000000],
        b: [0x00000001, 0x00000000, 0x00000000, 0x00000000],
        digest_6: [0x3B0F1AEB, 0x7FBAC531, 0xF7576FDB, 0xC4054075],
        digest_7: [0x43B017F6, 0x35DA5810, 0x1C2F8BF8, 0xB16F9B88],
    },
    SocketVector {
        name: "boundary", // gate-oracle
        a: [0x00000000, 0x00000001, 0xFFFFFFFE, 0xFFFFFFFF],
        b: [0x80000000, 0x7FFFFFFF, 0x00010000, 0x0000FFFF],
        digest_6: [0x03FA80ED, 0x79B929E7, 0x5FAF60C4, 0xB1F6E5C2],
        digest_7: [0x518B26CE, 0xDD289FC7, 0x5E623AAC, 0xD189075A],
    },
    SocketVector {
        name: "formula_0", // socket-kats
        a: [0x9E3779B9, 0x3C6EF372, 0xDAA66D2B, 0x78DDE6E4],
        b: [0x8FF34781, 0x2E2AC13A, 0xCC623AF3, 0x6A99B4AC],
        digest_6: [0x937642E5, 0x03E78AFE, 0x85C6D9CD, 0x42824862],
        digest_7: [0xAB67E603, 0x5D251992, 0xCC7CE527, 0xD4EF91ED],
    },
    SocketVector {
        name: "formula_1", // socket-kats
        a: [0x81AF1549, 0x1FE68F02, 0xBE1E08BB, 0x5C558274],
        b: [0x736AE311, 0x11A25CCA, 0xAFD9D683, 0x4E11503C],
        digest_6: [0xACE60EA5, 0x8135DE83, 0xEAE0B6DA, 0x9934A2F0],
        digest_7: [0x653C2C04, 0x78EF5846, 0x6A736E8A, 0x914D4DB9],
    },
    SocketVector {
        name: "formula_1", // gate-oracle
        a: [0x01020304, 0x05060708, 0x090A0B0C, 0x0D0E0F10],
        b: [0x11121314, 0x15161718, 0x191A1B1C, 0x1D1E1F20],
        digest_6: [0xF227539A, 0x2585193F, 0x16FB766E, 0x4B577722],
        digest_7: [0xADED3052, 0x697F82FA, 0x7AEA4B01, 0x04BFCDBC],
    },
    SocketVector {
        name: "formula_2", // socket-kats
        a: [0x6526B0D9, 0x035E2A92, 0xA195A44B, 0x3FCD1E04],
        b: [0x56E27EA1, 0xF519F85A, 0x93517213, 0x3188EBCC],
        digest_6: [0xC594FDD3, 0xBB338F05, 0x33C5A455, 0x2E9D6C14],
        digest_7: [0xE87B6708, 0xCE220AD3, 0x5B64ED4F, 0xD8D574C2],
    },
    SocketVector {
        name: "formula_2", // gate-oracle
        a: [0xDEADBEEF, 0xCAFEBABE, 0x8BADF00D, 0xFEEDFACE],
        b: [0x0BADC0DE, 0xD15EA5E5, 0xC0FFEE00, 0xBAAAAAAD],
        digest_6: [0x7636E806, 0x8AAB225F, 0x7F947CE5, 0xA73023AF],
        digest_7: [0xDBD49162, 0x7A22E380, 0x81E2ECB6, 0xF29F4F76],
    },
    SocketVector {
        name: "formula_3", // socket-kats
        a: [0x489E4C69, 0xE6D5C622, 0x850D3FDB, 0x2344B994],
        b: [0x3A5A1A31, 0xD89193EA, 0x76C90DA3, 0x1500875C],
        digest_6: [0x4C93E057, 0xA31827EB, 0xFDE7CB52, 0x027F1933],
        digest_7: [0x5754C2B6, 0x35C665F2, 0xFFB72630, 0xEC470985],
    },
    SocketVector {
        name: "formula_3", // gate-oracle
        a: [0x7F800001, 0x00000002, 0x80000000, 0x7FFFFFFF],
        b: [0x00FF00FF, 0xFF00FF00, 0x0F0F0F0F, 0xF0F0F0F0],
        digest_6: [0xFEB2EB59, 0xD8D6F7E1, 0xDC2774D3, 0x09A984FE],
        digest_7: [0x5213902B, 0xC0A84BF1, 0xA070DC22, 0x531B944C],
    },
    SocketVector {
        name: "formula_4", // socket-kats
        a: [0x2C15E7F9, 0xCA4D61B2, 0x6884DB6B, 0x06BC5524],
        b: [0x1DD1B5C1, 0xBC092F7A, 0x5A40A933, 0xF87822EC],
        digest_6: [0x312C20F4, 0x077F08FF, 0x0608FFAF, 0x70423FD2],
        digest_7: [0xF910DA3B, 0x5CCD211F, 0xB6D1E097, 0xA7304D4D],
    },
    SocketVector {
        name: "max_min", // gate-oracle
        a: [0xFFFFFFFF, 0x00000000, 0xFFFFFFFF, 0x00000000],
        b: [0x00000000, 0xFFFFFFFF, 0x00000000, 0xFFFFFFFF],
        digest_6: [0xDCF8F70F, 0x18645178, 0xE0842849, 0x97FEC771],
        digest_7: [0x5B047362, 0xE1662BCF, 0x385D410A, 0xFD185A9E],
    },
    SocketVector {
        name: "nibble_ramp/nibble_ramp", // socket-kats+gate-oracle
        a: [0x00000000, 0x11111111, 0x22222222, 0x33333333],
        b: [0x44444444, 0x55555555, 0x66666666, 0x77777777],
        digest_6: [0xA218F925, 0x7819A69C, 0xC14B82EC, 0x60E8A949],
        digest_7: [0x2E24E9F5, 0xA0D24A4B, 0x53909030, 0xB64285F3],
    },
    SocketVector {
        name: "zeros/zeros", // socket-kats+gate-oracle
        a: [0x00000000, 0x00000000, 0x00000000, 0x00000000],
        b: [0x00000000, 0x00000000, 0x00000000, 0x00000000],
        digest_6: [0x9E1DF680, 0x17A425B2, 0x890775EA, 0xE2C6E09F],
        digest_7: [0x9484C177, 0xBA11AECD, 0x45BB9F21, 0x031F727D],
    },
];
