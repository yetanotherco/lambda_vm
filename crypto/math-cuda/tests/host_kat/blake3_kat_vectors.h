// Known-answer vectors for the BLAKE3 device kernels, embedded rather than
// parsed at run time.
//
// Embedded on purpose: a test that reads its vectors from a file passes
// silently when the read finds nothing, which is a failure mode that has
// already happened once on this harness. A table cannot have a zero-vector
// run, and `main` asserts the counts below as well.
//
// This file is DATA. It is transcribed, never computed, and the two tables
// come from outside this crate — see each one's provenance note.
#pragma once
#include <cstdint>

// ---------------------------------------------------------------------------
// Table 1 — the OFFICIAL BLAKE3 test vectors, standard 7-round hash.
//
// Transcribed from `thoughts/blake3/blake3-oracle/official_test_vectors.json`
// (tracked in this repo, sourced from the BLAKE3 reference implementation).
// Only the cases with `input_len <= 64` appear: a message that fits one block
// of one chunk is a SINGLE compression, which is what the device function
// computes. Longer cases need the chunk tree and are not this kernel's job.
//
// The input for length N is the first N bytes of the repeating 251-byte
// sequence 0, 1, 2, ..., 250 — the generator the vector file specifies.
// ---------------------------------------------------------------------------
struct OfficialVector {
    uint32_t input_len;
    const char *hash_hex;  // the first 32 bytes of the extended output
};

inline constexpr int NUM_OFFICIAL_VECTORS = 11;
inline constexpr OfficialVector OFFICIAL_VECTORS[NUM_OFFICIAL_VECTORS] = {
    { 0, "af1349b9f5f9a1a6a0404dea36dcc9499bcb25c9adc112b7cc9a93cae41f3262"},
    { 1, "2d3adedff11b61f14c886e35afa036736dcd87a74d27b5c1510225d0f592e213"},
    { 2, "7b7015bb92cf0b318037702a6cdd81dee41224f734684c2c122cd6359cb1ee63"},
    { 3, "e1be4d7a8ab5560aa4199eea339849ba8e293d55ca0a81006726d184519e647f"},
    { 4, "f30f5ab28fe047904037f77b6da4fea1e27241c5d132638d8bedce9d40494f32"},
    { 5, "b40b44dfd97e7a84a996a91af8b85188c66c126940ba7aad2e7ae6b385402aa2"},
    { 6, "06c4e8ffb6872fad96f9aaca5eee1553eb62aed0ad7198cef42e87f6a616c844"},
    { 7, "3f8770f387faad08faa9d8414e9f449ac68e6ff0417f673f602a646a891419fe"},
    { 8, "2351207d04fc16ade43ccab08600939c7c1fa70a5c0aaca76063d04c3228eaeb"},
    {63, "e9bc37a594daad83be9470df7f7b3798297c3d834ce80ba85d6e207627b7db7b"},
    {64, "4eed7141ea4a5cd4b788606bd23f46e212af9cacebacdc7d1f4c6dc7f2511b98"},
};

// ---------------------------------------------------------------------------
// Table 2 — the ten canonical vectors, at BOTH round counts.
//
// Inputs and `out6` are transcribed from `CANONICAL_VECTORS`
// (now `crypto/crypto/src/hash/blake3/vectors.rs`), whose 6-round outputs came
// from #903's Python oracle rather than from any Rust code. `out7` is
// `CANONICAL_OUT_7ROUND` from the same file, itself pinned by the official crate.
//
// This is the point of the table: it gives the SIX-round arm a known-answer
// test whose expected values no implementation in this tree produced.
//
// ★ It ALSO does a job Table 1 structurally CANNOT, and the reason is not
// obvious enough to leave unwritten: the official-vector path hashes whole
// messages, so it only ever exercises `t = 0`. A compression with the counter
// split inverted (`v[12]` and `v[13]` transposed) reproduces the official
// vectors at every single-block length, and is caught only here — all ten of
// these vectors carry `t >= 2^32`. Measured, not assumed. This table is not
// redundant with Table 1 and must not be retired as "covered by the standard".
//
// ★ PROVENANCE, strengthened 2026-08-15 — the 6-round column is no longer
// pinned by Python alone. `thoughts/blake3/reference-impl/` holds UPSTREAM
// BLAKE3 1.8.5 with a 2 KB reviewable diff (`PARAMETERISATION.diff`) whose only
// functional edit replaces seven unrolled `round_fn` calls with a loop bounded
// by `BLAKE3_ROUNDS_PARAM`. Built at both round counts and run over these ten
// inputs, it reproduces `out6` AND `out7` 10/10, all 16 words. It is C rather
// than Python, upstream's own code rather than a transcription, and it encodes
// the message schedule as an INDEXED TABLE (`MSG_SCHEDULE[r]`) rather than as an
// in-place permutation between rounds — a structurally different expression of
// the same convention, so its agreement cross-validates the schedule instead of
// restating it. Rebuild with `thoughts/blake3/reference-impl/build.sh`: a ~1
// second C compile, no cargo, no GPU.
// ---------------------------------------------------------------------------
struct CanonicalVector {
    uint32_t h[8];
    uint32_t m[16];
    uint64_t t;
    uint32_t block_len;
    uint32_t flags;
    uint32_t out6[16];
    uint32_t out7[16];
};

inline constexpr int NUM_CANONICAL_VECTORS = 10;
inline constexpr CanonicalVector CANONICAL_VECTORS[NUM_CANONICAL_VECTORS] = {
    {
        {0xD82C07CDu, 0x6BAA9455u, 0x82E2E662u, 0x7A024204u,
          0xE87A1613u, 0x81332876u, 0x48268673u, 0xC17C6279u},
        {0xE6F4590Bu, 0x4F65D4D9u, 0xBAD640FBu, 0xAF19922Au,
          0x19C78DF4u, 0x6F25E2A2u, 0xE9BB17BCu, 0x7A1D5006u,
          0x42AF9FC3u, 0x03983CA8u, 0xDE1B372Au, 0xDED733E8u,
          0x9148624Fu, 0xF7B0B7D2u, 0x72AE2244u, 0xEECE328Bu},
        0xB4E1357D4A84EB03ull, 42u, 52u,
        {0xCED9D1FFu, 0xC248EEABu, 0xBD109B7Fu, 0x911B48F6u,
          0x923D62C0u, 0xD804903Fu, 0x5974223Eu, 0xAA4F0C80u,
          0xAD61007Fu, 0xB50B8DDBu, 0xE7372BE1u, 0x33D3D6C3u,
          0x42AA284Bu, 0xC5A25F28u, 0x79AC8370u, 0xB75F3915u},
        {0xEE79E5DCu, 0xEA647B8Cu, 0x964C097Eu, 0xE2F3383Au,
          0xFE2E6D00u, 0x78EE613Au, 0xC33C8572u, 0xCD444391u,
          0x0C890604u, 0xC3209591u, 0x45633FF8u, 0xCB171C6Au,
          0x760247AEu, 0xF6D0FC1Eu, 0xCD550F20u, 0xCD54BF83u},
    },
    {
        {0xC386BBC4u, 0x414C343Cu, 0x7311D8A3u, 0xA6CECC1Bu,
          0xC9E9C616u, 0x18072E8Cu, 0xD5F4B3B2u, 0x7204E52Du},
        {0xF1FD42A2u, 0xE6C3F339u, 0x07D4BEDCu, 0x8A9A021Eu,
          0x3BAB6C39u, 0x05805975u, 0xA46D6753u, 0xDC2574BDu,
          0xAB99254Au, 0x4DA98F1Du, 0xE1EA24C4u, 0x815A47C5u,
          0x08D6AF57u, 0xCC22AF58u, 0x2C4A3698u, 0x5FEC898Fu},
        0xC74803E31BA16215ull, 50u, 94u,
        {0xF2A972E9u, 0x81FDB8ECu, 0x40C50EBCu, 0x4BA1CAF9u,
          0x9EE9E930u, 0x6B1A16B2u, 0xE9156F47u, 0xA89FB436u,
          0xA2F616B3u, 0x12874C12u, 0x30768035u, 0xE01A17D9u,
          0xBEE5C17Cu, 0xD61C0BE0u, 0x3041FF46u, 0xDFB91125u},
        {0xD68593D0u, 0xDBC8157Au, 0xF6E1687Cu, 0x52A60555u,
          0xB56D418Au, 0x0CCBB863u, 0xADBFB51Eu, 0x8BF7D125u,
          0x75C23432u, 0xF484D7A6u, 0x06E85F4Au, 0x2771FE96u,
          0x00F6E24Du, 0x48368A3Eu, 0x04EE7E88u, 0x501D8539u},
    },
    {
        {0x0E7A269Fu, 0x15BA2BDDu, 0xD5E34124u, 0x4EE207F8u,
          0x9B1F282Eu, 0x9B575BD1u, 0xF30B94FAu, 0x0706A045u},
        {0x6148A86Fu, 0x8697BBD0u, 0x8F7D9B78u, 0x3C729578u,
          0x061B9030u, 0x533C9135u, 0x829E07B0u, 0xE4C11AB2u,
          0xCBF87544u, 0xC34C769Fu, 0x5A91C89Bu, 0xF63F23D0u,
          0xC1066932u, 0x87C56473u, 0x7D718D73u, 0xECC1CB63u},
        0x7604E4B4E73695C3ull, 58u, 124u,
        {0x5AA6B114u, 0xC9D6740Cu, 0x8738CAF4u, 0xAC5F4B72u,
          0x9FC6B9DEu, 0x3F2EFB8Fu, 0x8CB7A912u, 0xF497A285u,
          0x3D062266u, 0x7F22380Cu, 0xAFD468FAu, 0x122CBA80u,
          0x446B156Du, 0xB239D8C2u, 0xC3EAB2CFu, 0x775F2F92u},
        {0xBC92D7C4u, 0x56542092u, 0x3490E2CBu, 0x2E3328CDu,
          0x13E3746Fu, 0xA5B88E66u, 0x2B5FE530u, 0x92C7AD52u,
          0xFF502AE5u, 0x1F088FBFu, 0x9163752Fu, 0x8A0C8B4Du,
          0xB557B0E8u, 0xE76F23CBu, 0xD054C959u, 0x74813CFDu},
    },
    {
        {0x8B529B4Au, 0x9A9A80FDu, 0xD6645FA9u, 0x3BFD1D33u,
          0x79F248B0u, 0x268ECC45u, 0xA2863A7Fu, 0x85EF3430u},
        {0xBDC2AE99u, 0x10645D51u, 0x97524D6Au, 0xDD933160u,
          0xE0F9E038u, 0xEBCD1F5Eu, 0xEF829C88u, 0xE0FD67DDu,
          0x18F2C41Cu, 0x22CEDAFBu, 0x378C74DCu, 0x4D100D8Fu,
          0x95C76AB4u, 0x95918694u, 0xE779C470u, 0xEDCF6109u},
        0x92D3043AFCF249F3ull, 36u, 31u,
        {0xEED92FABu, 0x138D9358u, 0x915BFE3Cu, 0x13718B01u,
          0xB506E277u, 0xBE4007CDu, 0x35847E06u, 0xCE1C6896u,
          0x52FA01B5u, 0x4AA26AF8u, 0xB1078A61u, 0x2C517AEDu,
          0xA08867A0u, 0xEA6ECFEAu, 0x6D33D3B0u, 0xDC293166u},
        {0xCF4FB929u, 0x1DBADE2Au, 0x70E63AAFu, 0x2E0FFB48u,
          0x60123045u, 0x798AEAE8u, 0x5A911D30u, 0x15977C61u,
          0x6F7C8334u, 0x5EB0BCE2u, 0xAB240F17u, 0x66B7A3CDu,
          0xA9064E0Bu, 0x6AC4747Bu, 0x1206F62Bu, 0x9F3E91ECu},
    },
    {
        {0x3C6DA5D7u, 0x656412A9u, 0x27AC435Au, 0x11072231u,
          0xEAFF1A09u, 0xC3E1B258u, 0x8963DC6Eu, 0x1B2ED40Eu},
        {0xED6F0B09u, 0xCE80C4B0u, 0xCCEA2645u, 0x3184FF27u,
          0x4F5253A0u, 0xE14B0190u, 0x9B191BF4u, 0xABF4A07Cu,
          0x81862FC9u, 0x2D83A823u, 0x793D0E45u, 0x4CDCE7A6u,
          0xE8ABB93Fu, 0xE1DF8AF9u, 0x8224B122u, 0x69F85E31u},
        0x49C7B59B995253FDull, 57u, 41u,
        {0xCA00BDA3u, 0x84239A3Au, 0xE7C88E6Du, 0x33A8A3D6u,
          0x09DCD1CEu, 0xA1B10212u, 0xF48E1156u, 0x8F039915u,
          0x8A055EAAu, 0xFF5B11D5u, 0xB725085Bu, 0x2E1AB267u,
          0x6AE7323Du, 0xB2FF6FA8u, 0x7102C8A1u, 0x7561EB37u},
        {0xFF525F0Fu, 0xD892E3D2u, 0xFB566B40u, 0x3BDF4ED0u,
          0x78B961CDu, 0x9CB86B48u, 0x6AB54F3Du, 0x3EF5F695u,
          0xBD896ED8u, 0x6265AC08u, 0xF6695D78u, 0x9F3795EAu,
          0x943E0342u, 0xD1437B3Bu, 0x4F6BAF78u, 0x85DFD2C9u},
    },
    {
        {0x9F767C45u, 0xBDE5C099u, 0xF17FD374u, 0xA6233255u,
          0xE6A16A3Bu, 0x1CFB10F6u, 0x3F1F65A8u, 0x8B33E968u},
        {0x92EDCF45u, 0x377B9AA2u, 0x478C281Du, 0xC4069545u,
          0xCC11D357u, 0x9E115E4Bu, 0x206F5C66u, 0xDF1461AAu,
          0xFB7FF337u, 0xDF561D80u, 0x4A0FE75Du, 0xF6236BF2u,
          0x346C6E2Bu, 0xB0CDE917u, 0xE4CC4132u, 0x4C7D6DF0u},
        0x6A3753915C76F18Aull, 18u, 67u,
        {0x14A9F66Fu, 0x101BDFE8u, 0x9B0A50DDu, 0xEE4BB45Bu,
          0x7A914502u, 0x77B3486Bu, 0x59BFC114u, 0xA1AD2AFDu,
          0xC194DDE6u, 0x894EC54Du, 0xAD36C805u, 0x9018F3F5u,
          0x165AF5D8u, 0x3E85B598u, 0x78E76653u, 0xBB7A485Du},
        {0xD22912BBu, 0x627F992Cu, 0xE883AF5Du, 0x50E58A48u,
          0xF3D071C6u, 0xB20D47A4u, 0x29011151u, 0xFE50E232u,
          0x594B76A3u, 0x8706296Bu, 0x2C1D1E31u, 0x6A478D0Du,
          0x64004E61u, 0xA072DA1Eu, 0xAB3FCA42u, 0x09BB269Eu},
    },
    {
        {0xD26B9496u, 0x42F9A039u, 0x001D9A88u, 0x5F877031u,
          0xC527E279u, 0x45CF8AA4u, 0xCD4A5557u, 0xAE9AF169u},
        {0xAF895F5Bu, 0xD822E2F9u, 0x17D7AB26u, 0xCCDF540Bu,
          0xCE06294Du, 0x4A8B0188u, 0xF38D2E64u, 0x5C41D5C5u,
          0xE8D5B9E3u, 0x5C832A51u, 0x9A0C1B76u, 0x4DE8344Eu,
          0x96D2F9E0u, 0x8677A5F2u, 0xA9A967C1u, 0x323BBEAFu},
        0x390567C27BD6AA42ull, 26u, 3u,
        {0x32A6FF70u, 0xC30560BCu, 0xD1C777C8u, 0xF1871821u,
          0x7207AB54u, 0x9F5B83C7u, 0xB6561C5Du, 0x991E738Fu,
          0xB38B62B9u, 0x0EF6D156u, 0x994BECB1u, 0x09A85D0Eu,
          0x32221741u, 0xADA3CC5Fu, 0x5B654ED6u, 0x2A7A62B2u},
        {0xA101CEABu, 0x9232E0ECu, 0x2FE4B24Eu, 0x35F7F4FEu,
          0x61A5AB42u, 0xBE417503u, 0xEB740D5Eu, 0x8BB2FE96u,
          0xC6863DA9u, 0x1F31FF5Du, 0x5763EA12u, 0xDC862699u,
          0x1A60ADE2u, 0x9E3E6745u, 0xE3C8F87Eu, 0xD3EFB0EAu},
    },
    {
        {0x269E0D37u, 0xA6A3A450u, 0x892F902Bu, 0x81E74EF5u,
          0x099950D8u, 0x6F03675Au, 0x11E20B8Fu, 0x6CAD4A26u},
        {0xF29D0DA9u, 0x658CDA14u, 0xF9EBDACCu, 0xDBC496CBu,
          0x4A23D596u, 0x2E44158Bu, 0xA38FD547u, 0x5F557203u,
          0x34B9B5DFu, 0x506BF2EFu, 0x7403E430u, 0x4CBD87ADu,
          0xCB5C7427u, 0x3E7D1BFBu, 0x930D6EAFu, 0x86734721u},
        0x12BD4ACEFAECBD38ull, 53u, 42u,
        {0xA632AD45u, 0x12CE41F4u, 0xD21B2CBDu, 0x76795C62u,
          0x6BEC36C1u, 0xDAFAFCDEu, 0x53CA87B7u, 0x92E8465Bu,
          0x7B424F5Du, 0xE1E6AD7Fu, 0x753BA387u, 0xCCC50824u,
          0x69AEDF6Du, 0xBBBBF253u, 0x78D04883u, 0xF3F33689u},
        {0x318604BEu, 0x22A35843u, 0x6CA63195u, 0xA2E7E2F8u,
          0x48769A04u, 0xC462F1E3u, 0x5CF053C7u, 0xFD1EE629u,
          0x69366332u, 0x0ACC819Bu, 0xBBD2456Au, 0xF1DA9DB6u,
          0x4A7B7D68u, 0x6DD1A843u, 0x61555466u, 0xBDA36F28u},
    },
    {
        {0x3A096533u, 0xF658F7A7u, 0x205738D1u, 0xB46EE1DAu,
          0x15CEB3A1u, 0x359B1548u, 0xA4517D6Cu, 0x7589CA4Au},
        {0x74007CB4u, 0xD49D0AC1u, 0x16EDC5D4u, 0x685CA8AFu,
          0x4223AA56u, 0x10269470u, 0x60908405u, 0xA92D04A3u,
          0x56A3E957u, 0xB0F91306u, 0xE6C08269u, 0xF2306D4Au,
          0x31A06A7Cu, 0x9436D6F6u, 0xE18692E2u, 0xE0C99F3Eu},
        0x329911DA9FBD8735ull, 19u, 91u,
        {0x913B2AE1u, 0xC7F73082u, 0x45E1C023u, 0x6F1F3F82u,
          0x20AEE6F5u, 0xDAF21D94u, 0xF2C1E4AFu, 0xD4F7D4ACu,
          0x44A45F87u, 0xF4C40CE5u, 0x613E9B94u, 0x08CE53DEu,
          0x4FF07AA4u, 0x456BF2E2u, 0x2066EA7Fu, 0x3C5A654Bu},
        {0x87584719u, 0x15C73090u, 0x851C1A4Au, 0x99D21014u,
          0x821A82A8u, 0xC7307CD5u, 0x6797EFE2u, 0xCF38CEDFu,
          0x777C177Du, 0x202BE3EAu, 0x19421985u, 0x3176132Du,
          0x7BB8BC22u, 0x65C9804Bu, 0x22C68EA3u, 0x92504162u},
    },
    {
        {0x5F915EF0u, 0x237751AAu, 0x01A5BA50u, 0x80B65386u,
          0x14B044D7u, 0x61076DC3u, 0xB99DE255u, 0x283B73A6u},
        {0x3CEE5E2Cu, 0x1C670EA9u, 0x972651DAu, 0x4A8AA593u,
          0xAC9ABB0Cu, 0x35BB5C11u, 0x47FBB3B4u, 0xCF3C17E5u,
          0xE2EB17C8u, 0xE11E99FBu, 0x7DE0D208u, 0x0602FE0Cu,
          0x98CAE043u, 0x9425B3E2u, 0x33FB4B4Fu, 0x15607DF9u},
        0xEAEB999B8A2E547Eull, 64u, 21u,
        {0xF5EE9114u, 0x856CABB8u, 0x29BE2CF1u, 0x603BE91Cu,
          0x94A7DD0Eu, 0x28FC3E27u, 0xB64E2CC8u, 0x2D2C67FFu,
          0x69FAC1BAu, 0x0C949090u, 0xD68DE435u, 0xCE91A527u,
          0xE80C1815u, 0x6D44EFE6u, 0x87C7B175u, 0xD18A8B94u},
        {0xDC60D189u, 0xE6311F18u, 0x9DC3E078u, 0x304BB43Eu,
          0x5C616E7Du, 0xE168D00Fu, 0x2E197872u, 0x175B9188u,
          0x5A99C462u, 0xEF311A88u, 0xC61836FDu, 0x9FFD4DE3u,
          0x36AE4940u, 0x4D813D81u, 0x9B058DA9u, 0x9017D38Cu},
    },
};

// ---------------------------------------------------------------------------
// Table 3 — MULTI-BLOCK official BLAKE3 vectors, for the `Blake3Chain`
// construction rather than the bare compression function.
//
// Table 1 above stops at 64 bytes because a single compression is all it can
// check. The chain spans many blocks, and its flag schedule, its `block_len`
// handling and its chaining value are only exercised past the first block — so
// it needs vectors Table 1 cannot supply.
//
// ★ These are still the OFFICIAL vectors, not an oracle's. `Blake3Chain` over a
// message of at most one chunk (1024 bytes) IS `blake3::hash` — standard
// BLAKE3's first chunk is exactly this chain, and a one-chunk message has that
// chunk's output as its root (PA-PLAN §1.7.2, P1). So for every length here up
// to 1024 the published hash is a direct known-answer test for the device
// chain, with no oracle and no transcription of anything computed in this repo.
//
// Transcribed from `thoughts/blake3/blake3-oracle/official_test_vectors.json`
// (tracked), first 32 bytes of each case's `hash`. Input for length N is the
// first N bytes of the repeating 251-byte sequence 0, 1, ..., 250 — the same
// generator Table 1 uses.
//
// `agrees` marks whether the chain must MATCH the published hash. The 1025 and
// 2048 rows are the P3 negative control: past one chunk standard BLAKE3 starts
// chunk 1 with a reset chaining value and builds a tree, and this construction
// deliberately does not. Without them "we implement the single-chunk chain"
// would be unfalsifiable — the matching rows alone would pass identically if the
// whole chunk tree had been implemented instead.
//
// ★ THE BOUNDARY IS LOCATED, not sampled. All 35 official cases were swept
// (2026-08-15): agreement holds for every length up to and including 1024, and
// fails for every one of the 18 lengths >= 1025. Max agreeing 1024, min
// differing 1025 — the divergence sits exactly on the one-chunk edge, which is
// what P3 predicts. The 1024/1025 pair below is that boundary; the other rows
// cover the multi-block cases in between.
//
// The input generator `i % 251` was itself verified empirically rather than
// taken from the file's prose: it reproduces the `hash` field for 35/35 cases.
// Note it differs from the `(37i + 11) mod 256` generator Table 4 uses — the two
// tables come from different sources and do NOT share a message.
// ---------------------------------------------------------------------------
struct ChainVector {
    uint32_t input_len;
    const char *hash_hex;
    bool agrees;  // false = must DIFFER (past one chunk)
};

inline constexpr int NUM_CHAIN_VECTORS = 8;
inline constexpr ChainVector CHAIN_VECTORS[NUM_CHAIN_VECTORS] = {
    {   65, "de1e5fa0be70df6d2be8fffd0e99ceaa8eb6e8c93a63f2d8d1c30ecb6b263dee", true},
    {  127, "d81293fda863f008c09e92fc382a81f5a0b4a1251cba1634016a0f86a6bd640d", true},
    {  128, "f17e570564b26578c33bb7f44643f539624b05df1a76c81f30acd548c44b45ef", true},
    {  129, "683aaae9f3c5ba37eaaf072aed0f9e30bac0865137bae68b1fde4ca2aebdcb12", true},
    { 1023, "10108970eeda3eb932baac1428c7a2163b0e924c9a9e25b35bba72b28f70bd11", true},
    { 1024, "42214739f095a406f3fc83deb889744ac00df831c10daa55189b5d121c855af7", true},
    { 1025, "d00278ae47eb27b34faecf67b4fe263f82d5412916c1ffd97c8cb7fb814b8444", false},
    { 2048, "e776b6028c7cd22a4d0ba182a8bf62205d2ef576467e838ed6f2529b85fba24a", false},
};

// ---------------------------------------------------------------------------
// Table 4 — the committed 6-ROUND chain KAT.
//
// A byte-for-byte transcription of `CHAIN_KAT_6ROUND`
// (`crypto/crypto/src/hash/blake3/chain.rs:304`), which is the same table
// PA-PLAN §1.7.5 records. Message of length N is byte `i = 37i + 11 (mod 256)`.
//
// Provenance, and why it is worth more than a self-comparison: those digests
// were produced by #903's Python oracle
// (`thoughts/blake3/blake3-oracle/blake3_ref.py`, tracked), a full
// standard-BLAKE3 implementation with the round count as a parameter, whose
// 7-round arm reproduces the official package bit-for-bit. So at the round count
// the campaign actually ships, this pins the device chain against numbers no
// Rust and no CUDA in this tree computed.
//
// Duplicating the table here rather than sharing one copy is deliberate: this
// harness compiles as standalone C++ with no cargo and no Rust in the build, so
// there is nothing to share it with. A drift between the two copies is caught by
// the Rust-side `device_chain_matches_the_committed_table_at_six_rounds`, which
// reads the Rust constant directly.
// ---------------------------------------------------------------------------
struct ChainKat6Round {
    uint32_t input_len;
    const char *hash_hex;
};

inline constexpr int NUM_CHAIN_KAT_6ROUND = 12;
inline constexpr ChainKat6Round CHAIN_KAT_6ROUND[NUM_CHAIN_KAT_6ROUND] = {
    {    0, "3c3bbb1f335a31ea86464b651c0206fc81d33262ae00ea1a65f3d1d04afaefc9"},
    {    1, "2a50e45b8921f9efa008d9f39f7165600cf48a7f0e859c2122e3ccb6b9677ee5"},
    {   31, "c38bf62f506040b2600273778d281b8943621e2b8a9f59e2379f8fd7e5c85125"},
    {   63, "c373f51a5eb8b27ea05bb1f6f4e62e924ff4d8a279f0d05afa5cd519391d6389"},
    {   64, "5900a1e398bb2bf6d3ba7f1a29197b79c86b71ad2c2631f4ac736c82db043cb5"},
    {   65, "53953fcadc39b8623901af7b534f2f6933e312f50299331334e6c0a7c9dbc2be"},
    {  127, "9e0dd8168d199a04590c2cba439b270776e42715d518f68655e56692483e505e"},
    {  128, "5caffc8784e817bbba991b2108c26a3dfdf804245ef63ae1040a3c34f1b362ff"},
    {  192, "399d6b9adeb2f88450775f773e9dec08836c135713c2c5dd09f4ceceb0ed3888"},
    {  256, "fbcab3699a4959fa37190e98ca5142ddbc88330f2e7d12335db9c6c8881a0b87"},
    { 1024, "f395e7e2150363b6d200487515425b0204eea424072183b701176eccbe0ffe1b"},
    { 1088, "b4738ede77a6ec166ee97667118d4793cbf2b08b45aac7c6d52943b5d298c688"},
};
