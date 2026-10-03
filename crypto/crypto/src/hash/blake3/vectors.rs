//! The canonical known-answer vectors for [`super::blake3_compress_rounds`],
//! at both round counts.
//!
//! Provenance is recorded in the parent module's header: the 7-round table is
//! what the official `blake3` crate produces (and is checked against it
//! directly), and the 6-round table came from #903's Python oracle, whose
//! conventions the 7-round arm pins from outside.

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

/// The same ten inputs as [`CANONICAL_VECTORS`], at **7 rounds** — that is,
/// under standard BLAKE3's compression function.
///
/// Provenance, and it is a rung stronger than the 6-round table's: these were
/// emitted by the gate-oracle's independently-written Python reference
/// (`thoughts/shared/lfm-real-hash/gate-oracle/blake3_oracle.py`) at
/// `rounds = 7` and cross-checked word-for-word against the second in-repo
/// reference (`thoughts/blake3/blake3-oracle/blake3_ref.py`) — two
/// implementations, agreeing on all ten. Both references' 7-round paths are
/// themselves pinned by the OFFICIAL BLAKE3 test vectors, so unlike
/// [`CANONICAL_VECTORS`] this table has an external anchor rather than one a
/// step removed. The same generation run re-derived the 6-round table and
/// reproduced it 10/10, which is what ties the two together.
///
/// Only the outputs are stored: the inputs are [`CANONICAL_VECTORS`]'s, and
/// duplicating them would be a second place for them to drift.
pub const CANONICAL_OUT_7ROUND: [[u32; 16]; 10] = [
    [
        0xEE79E5DC, 0xEA647B8C, 0x964C097E, 0xE2F3383A, 0xFE2E6D00, 0x78EE613A, 0xC33C8572,
        0xCD444391, 0x0C890604, 0xC3209591, 0x45633FF8, 0xCB171C6A, 0x760247AE, 0xF6D0FC1E,
        0xCD550F20, 0xCD54BF83,
    ],
    [
        0xD68593D0, 0xDBC8157A, 0xF6E1687C, 0x52A60555, 0xB56D418A, 0x0CCBB863, 0xADBFB51E,
        0x8BF7D125, 0x75C23432, 0xF484D7A6, 0x06E85F4A, 0x2771FE96, 0x00F6E24D, 0x48368A3E,
        0x04EE7E88, 0x501D8539,
    ],
    [
        0xBC92D7C4, 0x56542092, 0x3490E2CB, 0x2E3328CD, 0x13E3746F, 0xA5B88E66, 0x2B5FE530,
        0x92C7AD52, 0xFF502AE5, 0x1F088FBF, 0x9163752F, 0x8A0C8B4D, 0xB557B0E8, 0xE76F23CB,
        0xD054C959, 0x74813CFD,
    ],
    [
        0xCF4FB929, 0x1DBADE2A, 0x70E63AAF, 0x2E0FFB48, 0x60123045, 0x798AEAE8, 0x5A911D30,
        0x15977C61, 0x6F7C8334, 0x5EB0BCE2, 0xAB240F17, 0x66B7A3CD, 0xA9064E0B, 0x6AC4747B,
        0x1206F62B, 0x9F3E91EC,
    ],
    [
        0xFF525F0F, 0xD892E3D2, 0xFB566B40, 0x3BDF4ED0, 0x78B961CD, 0x9CB86B48, 0x6AB54F3D,
        0x3EF5F695, 0xBD896ED8, 0x6265AC08, 0xF6695D78, 0x9F3795EA, 0x943E0342, 0xD1437B3B,
        0x4F6BAF78, 0x85DFD2C9,
    ],
    [
        0xD22912BB, 0x627F992C, 0xE883AF5D, 0x50E58A48, 0xF3D071C6, 0xB20D47A4, 0x29011151,
        0xFE50E232, 0x594B76A3, 0x8706296B, 0x2C1D1E31, 0x6A478D0D, 0x64004E61, 0xA072DA1E,
        0xAB3FCA42, 0x09BB269E,
    ],
    [
        0xA101CEAB, 0x9232E0EC, 0x2FE4B24E, 0x35F7F4FE, 0x61A5AB42, 0xBE417503, 0xEB740D5E,
        0x8BB2FE96, 0xC6863DA9, 0x1F31FF5D, 0x5763EA12, 0xDC862699, 0x1A60ADE2, 0x9E3E6745,
        0xE3C8F87E, 0xD3EFB0EA,
    ],
    [
        0x318604BE, 0x22A35843, 0x6CA63195, 0xA2E7E2F8, 0x48769A04, 0xC462F1E3, 0x5CF053C7,
        0xFD1EE629, 0x69366332, 0x0ACC819B, 0xBBD2456A, 0xF1DA9DB6, 0x4A7B7D68, 0x6DD1A843,
        0x61555466, 0xBDA36F28,
    ],
    [
        0x87584719, 0x15C73090, 0x851C1A4A, 0x99D21014, 0x821A82A8, 0xC7307CD5, 0x6797EFE2,
        0xCF38CEDF, 0x777C177D, 0x202BE3EA, 0x19421985, 0x3176132D, 0x7BB8BC22, 0x65C9804B,
        0x22C68EA3, 0x92504162,
    ],
    [
        0xDC60D189, 0xE6311F18, 0x9DC3E078, 0x304BB43E, 0x5C616E7D, 0xE168D00F, 0x2E197872,
        0x175B9188, 0x5A99C462, 0xEF311A88, 0xC61836FD, 0x9FFD4DE3, 0x36AE4940, 0x4D813D81,
        0x9B058DA9, 0x9017D38C,
    ],
];
