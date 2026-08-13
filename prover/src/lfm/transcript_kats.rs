//! Transcript KATs for the LFM compress-chain Fiat–Shamir transcript, at 6 and
//! 7 rounds.
//!
//! GENERATED — do not hand-edit. The INPUTS come from
//! `thoughts/shared/lfm-real-hash/transcript-spec/transcript_kats.json`, which
//! the oracle produced from a Python reference written **before any Rust
//! existed**. That ordering is the point: these vectors are a specification the
//! implementation is checked against, not a recording of what the
//! implementation happened to do.
//!
//! ⚠ **The results were re-pinned when the socket widened to twelve lanes**, by
//! `leaf-spec/rate4_kat_gen.py` out of the same oracle. All 12 moved and no
//! input did: `block_len` is `v[14]` and cannot be made mode-dependent, so the
//! transcript domain re-blesses alongside the leaf domain that needed the width
//! (COMMIT.md §1.4.4 H9).
//!
//! Framing (transcript spec §1.2 at the COMMIT.md §1.2 width): identical to the
//! Merkle socket in every respect except the tag word, which is `"LFMT"`
//! instead of `"LFMC"`. So h = BLAKE3_IV, m[0..4] = state, m[4..8] = operand,
//! m[8..12] = 0 — the third input cell, which the unread-`IN` pins force to
//! zero — m[12] = "LFMT", m[13..16] = 0, t = 0, block_len = 52, flags = 0x0B,
//! digest = out[0..4]. At 7 rounds a step is still literally
//! `blake3::hash(state ‖ operand ‖ 0^16 ‖ "LFMT")[..16]`.

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
        result_6: [0x58A784C6, 0xCA20122A, 0x574D1385, 0x4C7F61AC],
        result_7: [0xBB5DF0AD, 0xBB660FC6, 0x401C1FAD, 0x651C297C],
    },
    StepVector {
        name: "zero_state_main_root",
        state: [0x00000000, 0x00000000, 0x00000000, 0x00000000],
        operand: [0x01020304, 0x05060708, 0x090A0B0C, 0x0D0E0F10],
        result_6: [0xBFD9E2ED, 0x726EDE27, 0x91805DE1, 0xC11F0DA8],
        result_7: [0x503FDDF4, 0x48633531, 0x8EEA401C, 0x213213C8],
    },
    StepVector {
        name: "ramp_state_ramp_operand",
        state: [0x01020304, 0x05060708, 0x090A0B0C, 0x0D0E0F10],
        operand: [0x11121314, 0x15161718, 0x191A1B1C, 0x1D1E1F20],
        result_6: [0x00A8B31B, 0x0C48A09A, 0x1D06A9A8, 0x6C27BD61],
        result_7: [0x29D95598, 0x69E4FD73, 0x243BFCE9, 0x14598F96],
    },
    StepVector {
        name: "max_state",
        state: [0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF],
        operand: [0xDEADBEEF, 0xCAFEBABE, 0x8BADF00D, 0xFEEDFACE],
        result_6: [0xA710A43E, 0x62E96839, 0xE00D7CA2, 0x1E054FEF],
        result_7: [0xBA98C5EF, 0xAFDC8C3E, 0xFA425A12, 0xF35C1B47],
    },
    StepVector {
        name: "squeeze_operand_0",
        state: [0x01020304, 0x05060708, 0x090A0B0C, 0x0D0E0F10],
        operand: [0x305A5153, 0x00000000, 0x00000000, 0x00000000],
        result_6: [0xE1F1E0DF, 0x9B1491E4, 0x26F46CE4, 0x644BA9F0],
        result_7: [0x2EBCFDA8, 0x2F7C4E72, 0xAE841641, 0x6751FE80],
    },
    StepVector {
        name: "squeeze_operand_255",
        state: [0x11121314, 0x15161718, 0x191A1B1C, 0x1D1E1F20],
        operand: [0x305A5153, 0x000000FF, 0x00000000, 0x00000000],
        result_6: [0xDD47DC57, 0x9AC95714, 0xE774A0DA, 0xD4703C0B],
        result_7: [0x33396F61, 0x832BA04F, 0x2BB788AB, 0xE9FE006B],
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
        [0x503FDDF4, 0x48633531, 0x8EEA401C, 0x213213C8],
        [0x02FEA9A6, 0xE6BF885C, 0xF174E65F, 0x4EC9AB10],
        [0xF91F56DE, 0x62F37956, 0xE67D5421, 0xD82727D0],
        [0x7A31B840, 0xAD7F2625, 0xE27D1C56, 0xCDB0E9A7],
        [0xA6790B7B, 0x00695D49, 0xA663DC33, 0x2E849F0C],
        [0xFC40465E, 0x0092B147, 0x2FA48645, 0x9755608B],
        [0x6BD6F0B0, 0x634CF1C6, 0x3CBD9D2D, 0x349F278B],
        [0xD86E1D3F, 0xDD1CFBC3, 0x1C8E8F14, 0x22D35494],
        [0x5B449138, 0x3435B7D5, 0x7CFE4C06, 0x1C022FCF],
        [0xE8D9848C, 0x0429B6F7, 0xDD5CBA1A, 0xBD465F16],
        [0xD1E3472D, 0xD945386A, 0xC746A9B3, 0x8FD73C31],
    ],
    alpha: [0x503FDDF4, 0x48633531, 0x8EEA401C],
    zeta0: [0x02FEA9A6, 0xE6BF885C, 0xF174E65F],
    zeta1: [0x7A31B840, 0xAD7F2625, 0xE27D1C56],
    query_bits: [[0, 0, 0, 0], [1, 1, 1, 1], [0, 0, 0, 1], [0, 0, 1, 1]],
};

/// The end-to-end vector at 6 rounds (`--features blake3-6round`).
pub const FRI_TOY_6: EndToEndVector = EndToEndVector {
    states: [
        [0xBFD9E2ED, 0x726EDE27, 0x91805DE1, 0xC11F0DA8],
        [0x9DDE5E52, 0xD7018142, 0x978528FF, 0xA01782B7],
        [0x189BAD31, 0x364E3C30, 0x6B4D5516, 0x78D7FE7B],
        [0x8E3E80BB, 0xFCA1AF96, 0xA59E0F41, 0x18C9AB19],
        [0x4DF9BD75, 0x2E131E7D, 0x2DEB348E, 0x62BC30F3],
        [0x202DB868, 0x8FF72AC4, 0x452D536A, 0x78DECD7B],
        [0xA3D1E95E, 0x8C68ECF6, 0x91D7DF5E, 0x28D8CBB2],
        [0xFA4222A2, 0x18DA7862, 0x3BBDA144, 0xAB453013],
        [0x525CE059, 0x0E9B9EAB, 0xFB57633B, 0xE43490A7],
        [0x261309AA, 0x84C675B7, 0xD5BBF0EB, 0x4AF40850],
        [0x4D62F602, 0x2CC5D660, 0x99C44AF7, 0xEB502D8A],
    ],
    alpha: [0xBFD9E2ED, 0x726EDE27, 0x91805DE1],
    zeta0: [0x9DDE5E52, 0xD7018142, 0x978528FF],
    zeta1: [0x8E3E80BB, 0xFCA1AF96, 0xA59E0F41],
    query_bits: [[0, 1, 1, 1], [0, 1, 0, 0], [1, 0, 0, 1], [0, 1, 0, 1]],
};
