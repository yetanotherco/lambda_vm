//! Transcript KATs for the LFM compress-chain Fiat–Shamir transcript, at 6 and
//! 7 rounds.
//!
//! GENERATED — do not hand-edit. Rendered from
//! `thoughts/shared/lfm-real-hash/transcript-spec/transcript_kats.json`, which
//! the oracle produced from a Python reference written **before any Rust
//! existed**. That ordering is the point: these vectors are a specification the
//! implementation is checked against, not a recording of what the
//! implementation happened to do.
//!
//! Framing (transcript spec §1.2): identical to the Merkle socket in every
//! respect except `m[8]`, which is `"LFMT"` instead of `"LFMC"`. So h =
//! BLAKE3_IV, m[0..4] = state, m[4..8] = operand, m[9..16] = 0, t = 0,
//! block_len = 36, flags = 0x0B, digest = out[0..4] — and at 7 rounds a step is
//! literally `blake3::hash(state ‖ operand ‖ "LFMT")[..16]`.

/// One transcript step: state, operand, and the resulting state at each round
/// count.
pub struct StepVector {
    pub name: &'static str,
    pub state: [u32; 4],
    pub operand: [u32; 4],
    /// Result at 6 rounds (the A6R variant; no library computes it).
    pub result_6: [u32; 4],
    /// Result at 7 rounds — `blake3::hash(state ‖ operand ‖ "LFMT")[..16]`.
    pub result_7: [u32; 4],
}

pub const STEP_VECTORS: [StepVector; 6] = [
    StepVector {
        name: "zero_state_zero_operand",
        state: [0x00000000, 0x00000000, 0x00000000, 0x00000000],
        operand: [0x00000000, 0x00000000, 0x00000000, 0x00000000],
        result_6: [0xC072FE26, 0x3B4C920F, 0x64BD29A0, 0x0213E6E4],
        result_7: [0xE1DDB56C, 0x1454CCCA, 0xB008D630, 0x4537F7A3],
    },
    StepVector {
        name: "zero_state_main_root",
        state: [0x00000000, 0x00000000, 0x00000000, 0x00000000],
        operand: [0x01020304, 0x05060708, 0x090A0B0C, 0x0D0E0F10],
        result_6: [0x8A9AE283, 0xC782CB0F, 0x257502C4, 0x713479FF],
        result_7: [0xD3FD9F50, 0x3ED183D9, 0xF60EE882, 0xE3C34674],
    },
    StepVector {
        name: "ramp_state_ramp_operand",
        state: [0x01020304, 0x05060708, 0x090A0B0C, 0x0D0E0F10],
        operand: [0x11121314, 0x15161718, 0x191A1B1C, 0x1D1E1F20],
        result_6: [0x233B6A30, 0xC0988F42, 0x12354C22, 0x589508FB],
        result_7: [0x6D6995B4, 0xFA62C580, 0x17872A49, 0x2C4E04D1],
    },
    StepVector {
        name: "max_state",
        state: [0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF],
        operand: [0xDEADBEEF, 0xCAFEBABE, 0x8BADF00D, 0xFEEDFACE],
        result_6: [0xCC7D56B3, 0xCCCA9F29, 0x0239B3EC, 0x3EE001E6],
        result_7: [0x6B2F25BB, 0x1D0F16EC, 0x1F75DC73, 0xB96320BF],
    },
    StepVector {
        name: "squeeze_operand_0",
        state: [0x01020304, 0x05060708, 0x090A0B0C, 0x0D0E0F10],
        operand: [0x305A5153, 0x00000000, 0x00000000, 0x00000000],
        result_6: [0x37371DD1, 0x75B3F42F, 0xFA61B49C, 0xECA8FBF8],
        result_7: [0x257B36C1, 0x52274AF7, 0xA650F1CF, 0xDAC13C51],
    },
    StepVector {
        name: "squeeze_operand_255",
        state: [0x11121314, 0x15161718, 0x191A1B1C, 0x1D1E1F20],
        operand: [0x305A5153, 0x000000FF, 0x00000000, 0x00000000],
        result_6: [0x634D0599, 0xFAAD44C3, 0x9298BDC4, 0x157B8CCB],
        result_7: [0x1AFC8DC4, 0x04B3C139, 0xEB73F81F, 0x48083394],
    },
];

/// The END-TO-END vector: a `FriToyV0`-preamble-shaped transcript, op by op.
///
/// The operation sequence is fixed and lives in the test that replays it —
/// `absorb(main_root), squeeze, squeeze, absorb(l1_root), squeeze,
/// absorb_felts(t0w), absorb_felts(t1w), 4× squeeze_bits`, ✓ VERIFIED against
/// `programs::fri_toy_program_source`. What is pinned here is the STATE after
/// every recorded op, so a divergence is located at the step it happened rather
/// than at the end.
///
/// The last two absorbs are `absorb_felts`, not `absorb2`: the terminal
/// coefficients are field DATA, so each is leaf-hashed under `"LFML"` and the
/// DIGEST is absorbed. The transcript's step count is the same either way, which
/// is why this vector had to be re-pointed deliberately when the program moved
/// rather than caught by a red test.
pub struct EndToEndVector {
    /// State after each recorded op, in order.
    pub states: [[u32; 4]; 11],
    /// The three ext challenges (lanes 0–2 of a squeezed cell).
    pub alpha: [u32; 3],
    pub zeta0: [u32; 3],
    pub zeta1: [u32; 3],
    /// `QUERY_BITS` index bits per query, low-to-high.
    pub query_bits: [[u8; 4]; 4],
}

/// The transcript's inputs — the four cells the preamble absorbs.
pub const MAIN_ROOT: [u32; 4] = [0x01020304, 0x05060708, 0x090A0B0C, 0x0D0E0F10];

pub const L1_ROOT: [u32; 4] = [0x11121314, 0x15161718, 0x191A1B1C, 0x1D1E1F20];

pub const T0W: [u32; 4] = [0xDEADBEEF, 0xCAFEBABE, 0x8BADF00D, 0xFEEDFACE];

pub const T1W: [u32; 4] = [0x0BADC0DE, 0xD15EA5E5, 0xC0FFEE00, 0xBAAAAAAD];

/// Compressions the whole preamble costs — the oracle's cost claim.
///
/// **13, not 11.** Eleven are TRANSCRIPT steps (5 absorbs, 6 squeezes); the
/// other two are the LEAF rows `absorb_felts` adds, one per data cell. Counting
/// them here is the oracle's convention and it is the one that closes the
/// `FriToyV0` total: 4 queries × 20 + 13 = 93.
pub const FRI_TOY_COMPRESSIONS: usize = 13;

/// The end-to-end vector at 7 rounds (the default build).
pub const FRI_TOY_7: EndToEndVector = EndToEndVector {
    states: [
        [0xD3FD9F50, 0x3ED183D9, 0xF60EE882, 0xE3C34674],
        [0x27023F83, 0xA1344FB0, 0x9EBDBBB2, 0x00158D9B],
        [0x43FFB960, 0x3696C76D, 0x9D106062, 0xEAA3E925],
        [0x23D1D389, 0x3FE9FBB1, 0x7AF56AE7, 0xEC936F39],
        [0x94153DE2, 0xA6003377, 0xD028ED4B, 0xF3EB8582],
        [0xEC821701, 0xCD13E17E, 0x7EADC68F, 0x01E38C58],
        [0x0E8226D7, 0x1E2E2338, 0x845CF387, 0xE33EBDEC],
        [0xB654D354, 0x71EDED11, 0x8AFF36B2, 0xA6C750AF],
        [0x06A07FAD, 0x8CA90A52, 0x7A48DF49, 0xC9C1AED8],
        [0x8B5EA0EF, 0xF22C1FA1, 0xE1BA9F92, 0xD20CB729],
        [0xA25C2860, 0xFECF62F7, 0x72A5F0EF, 0x0F2BE133],
    ],
    alpha: [0xD3FD9F50, 0x3ED183D9, 0xF60EE882],
    zeta0: [0x27023F83, 0xA1344FB0, 0x9EBDBBB2],
    zeta1: [0x23D1D389, 0x3FE9FBB1, 0x7AF56AE7],
    query_bits: [[1, 1, 1, 0], [0, 0, 1, 0], [1, 0, 1, 1], [1, 1, 1, 1]],
};

/// The end-to-end vector at 6 rounds (`--features blake3-6round`).
pub const FRI_TOY_6: EndToEndVector = EndToEndVector {
    states: [
        [0x8A9AE283, 0xC782CB0F, 0x257502C4, 0x713479FF],
        [0x88D30EFA, 0xCE8D4E24, 0xA3049DB6, 0x93341D6F],
        [0x0953D5A3, 0x4D25B331, 0x4B1A3E0A, 0x6D7D710E],
        [0x408B335E, 0xFB12033E, 0x4ED4D8F5, 0x6077EE28],
        [0xB8746B5E, 0x99C839BC, 0x74F64FED, 0x81FB37FF],
        [0xBEA3BF5F, 0x44DA486A, 0x2876E758, 0xB22EA9D0],
        [0x91239994, 0x05C16E77, 0x4DF175AF, 0xC74094E4],
        [0x3997959E, 0x3EF54A2F, 0xD791B584, 0x6AC75C52],
        [0x384A95CE, 0x6CB0B223, 0x6A50D4CB, 0xA38A6D79],
        [0x5C4AC682, 0x9DEEE8F9, 0xB0752A41, 0xF87991A2],
        [0xF71FF60F, 0xDB60DF57, 0x188420D7, 0xF2A6C54A],
    ],
    alpha: [0x8A9AE283, 0xC782CB0F, 0x257502C4],
    zeta0: [0x88D30EFA, 0xCE8D4E24, 0xA3049DB6],
    zeta1: [0x408B335E, 0xFB12033E, 0x4ED4D8F5],
    query_bits: [[0, 0, 1, 0], [0, 1, 1, 1], [0, 1, 1, 1], [0, 1, 0, 0]],
};
