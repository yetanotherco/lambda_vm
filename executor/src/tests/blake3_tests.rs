//! Tests for the BLAKE3 6-round compression and its accelerator syscall.
//!
//! Ground truth is the validated oracle (`thoughts/blake3/blake3-oracle/`):
//! the pinned canonical 6-round vectors below were emitted by its harness and
//! checked against the official `blake3` crate at the official test-vector
//! parameters. The `t` values exercise the full 64-bit counter range, pinning
//! the `t_lo → v[12]` / `t_hi → v[13]` split order.

use crate::vm::instruction::decoding::Instruction;
use crate::vm::instruction::execution::{
    BLAKE3_ABSORB_MAX_BLOCKS, BLAKE3_ABSORB_SYSCALL_NUMBER, BLAKE3_SYSCALL_NUMBER, ExecutionError,
    blake3_compress_6round,
};
use crate::vm::memory::Memory;
use crate::vm::registers::Registers;

/// One pinned 6-round vector from the validated oracle.
struct Blake3Vector {
    h: [u32; 8],
    m: [u32; 16],
    t: u64,
    block_len: u32,
    flags: u32,
    out: [u32; 16],
}

/// The 10 canonical 6-round vectors, generated from
/// `thoughts/blake3/blake3-oracle/canonical_6round_vectors.json` (which the
/// oracle harness regenerates and which was validated against the official
/// `blake3` crate). Do not edit by hand.
const CANONICAL_6ROUND_VECTORS: &[Blake3Vector] = &[
    Blake3Vector {
        h: [
            0xd82c07cd, 0x6baa9455, 0x82e2e662, 0x7a024204, 0xe87a1613, 0x81332876, 0x48268673,
            0xc17c6279,
        ],
        m: [
            0xe6f4590b, 0x4f65d4d9, 0xbad640fb, 0xaf19922a, 0x19c78df4, 0x6f25e2a2, 0xe9bb17bc,
            0x7a1d5006, 0x42af9fc3, 0x03983ca8, 0xde1b372a, 0xded733e8, 0x9148624f, 0xf7b0b7d2,
            0x72ae2244, 0xeece328b,
        ],
        t: 0xb4e1357d4a84eb03,
        block_len: 42,
        flags: 52,
        out: [
            0xced9d1ff, 0xc248eeab, 0xbd109b7f, 0x911b48f6, 0x923d62c0, 0xd804903f, 0x5974223e,
            0xaa4f0c80, 0xad61007f, 0xb50b8ddb, 0xe7372be1, 0x33d3d6c3, 0x42aa284b, 0xc5a25f28,
            0x79ac8370, 0xb75f3915,
        ],
    },
    Blake3Vector {
        h: [
            0xc386bbc4, 0x414c343c, 0x7311d8a3, 0xa6cecc1b, 0xc9e9c616, 0x18072e8c, 0xd5f4b3b2,
            0x7204e52d,
        ],
        m: [
            0xf1fd42a2, 0xe6c3f339, 0x07d4bedc, 0x8a9a021e, 0x3bab6c39, 0x05805975, 0xa46d6753,
            0xdc2574bd, 0xab99254a, 0x4da98f1d, 0xe1ea24c4, 0x815a47c5, 0x08d6af57, 0xcc22af58,
            0x2c4a3698, 0x5fec898f,
        ],
        t: 0xc74803e31ba16215,
        block_len: 50,
        flags: 94,
        out: [
            0xf2a972e9, 0x81fdb8ec, 0x40c50ebc, 0x4ba1caf9, 0x9ee9e930, 0x6b1a16b2, 0xe9156f47,
            0xa89fb436, 0xa2f616b3, 0x12874c12, 0x30768035, 0xe01a17d9, 0xbee5c17c, 0xd61c0be0,
            0x3041ff46, 0xdfb91125,
        ],
    },
    Blake3Vector {
        h: [
            0x0e7a269f, 0x15ba2bdd, 0xd5e34124, 0x4ee207f8, 0x9b1f282e, 0x9b575bd1, 0xf30b94fa,
            0x0706a045,
        ],
        m: [
            0x6148a86f, 0x8697bbd0, 0x8f7d9b78, 0x3c729578, 0x061b9030, 0x533c9135, 0x829e07b0,
            0xe4c11ab2, 0xcbf87544, 0xc34c769f, 0x5a91c89b, 0xf63f23d0, 0xc1066932, 0x87c56473,
            0x7d718d73, 0xecc1cb63,
        ],
        t: 0x7604e4b4e73695c3,
        block_len: 58,
        flags: 124,
        out: [
            0x5aa6b114, 0xc9d6740c, 0x8738caf4, 0xac5f4b72, 0x9fc6b9de, 0x3f2efb8f, 0x8cb7a912,
            0xf497a285, 0x3d062266, 0x7f22380c, 0xafd468fa, 0x122cba80, 0x446b156d, 0xb239d8c2,
            0xc3eab2cf, 0x775f2f92,
        ],
    },
    Blake3Vector {
        h: [
            0x8b529b4a, 0x9a9a80fd, 0xd6645fa9, 0x3bfd1d33, 0x79f248b0, 0x268ecc45, 0xa2863a7f,
            0x85ef3430,
        ],
        m: [
            0xbdc2ae99, 0x10645d51, 0x97524d6a, 0xdd933160, 0xe0f9e038, 0xebcd1f5e, 0xef829c88,
            0xe0fd67dd, 0x18f2c41c, 0x22cedafb, 0x378c74dc, 0x4d100d8f, 0x95c76ab4, 0x95918694,
            0xe779c470, 0xedcf6109,
        ],
        t: 0x92d3043afcf249f3,
        block_len: 36,
        flags: 31,
        out: [
            0xeed92fab, 0x138d9358, 0x915bfe3c, 0x13718b01, 0xb506e277, 0xbe4007cd, 0x35847e06,
            0xce1c6896, 0x52fa01b5, 0x4aa26af8, 0xb1078a61, 0x2c517aed, 0xa08867a0, 0xea6ecfea,
            0x6d33d3b0, 0xdc293166,
        ],
    },
    Blake3Vector {
        h: [
            0x3c6da5d7, 0x656412a9, 0x27ac435a, 0x11072231, 0xeaff1a09, 0xc3e1b258, 0x8963dc6e,
            0x1b2ed40e,
        ],
        m: [
            0xed6f0b09, 0xce80c4b0, 0xccea2645, 0x3184ff27, 0x4f5253a0, 0xe14b0190, 0x9b191bf4,
            0xabf4a07c, 0x81862fc9, 0x2d83a823, 0x793d0e45, 0x4cdce7a6, 0xe8abb93f, 0xe1df8af9,
            0x8224b122, 0x69f85e31,
        ],
        t: 0x49c7b59b995253fd,
        block_len: 57,
        flags: 41,
        out: [
            0xca00bda3, 0x84239a3a, 0xe7c88e6d, 0x33a8a3d6, 0x09dcd1ce, 0xa1b10212, 0xf48e1156,
            0x8f039915, 0x8a055eaa, 0xff5b11d5, 0xb725085b, 0x2e1ab267, 0x6ae7323d, 0xb2ff6fa8,
            0x7102c8a1, 0x7561eb37,
        ],
    },
    Blake3Vector {
        h: [
            0x9f767c45, 0xbde5c099, 0xf17fd374, 0xa6233255, 0xe6a16a3b, 0x1cfb10f6, 0x3f1f65a8,
            0x8b33e968,
        ],
        m: [
            0x92edcf45, 0x377b9aa2, 0x478c281d, 0xc4069545, 0xcc11d357, 0x9e115e4b, 0x206f5c66,
            0xdf1461aa, 0xfb7ff337, 0xdf561d80, 0x4a0fe75d, 0xf6236bf2, 0x346c6e2b, 0xb0cde917,
            0xe4cc4132, 0x4c7d6df0,
        ],
        t: 0x6a3753915c76f18a,
        block_len: 18,
        flags: 67,
        out: [
            0x14a9f66f, 0x101bdfe8, 0x9b0a50dd, 0xee4bb45b, 0x7a914502, 0x77b3486b, 0x59bfc114,
            0xa1ad2afd, 0xc194dde6, 0x894ec54d, 0xad36c805, 0x9018f3f5, 0x165af5d8, 0x3e85b598,
            0x78e76653, 0xbb7a485d,
        ],
    },
    Blake3Vector {
        h: [
            0xd26b9496, 0x42f9a039, 0x001d9a88, 0x5f877031, 0xc527e279, 0x45cf8aa4, 0xcd4a5557,
            0xae9af169,
        ],
        m: [
            0xaf895f5b, 0xd822e2f9, 0x17d7ab26, 0xccdf540b, 0xce06294d, 0x4a8b0188, 0xf38d2e64,
            0x5c41d5c5, 0xe8d5b9e3, 0x5c832a51, 0x9a0c1b76, 0x4de8344e, 0x96d2f9e0, 0x8677a5f2,
            0xa9a967c1, 0x323bbeaf,
        ],
        t: 0x390567c27bd6aa42,
        block_len: 26,
        flags: 3,
        out: [
            0x32a6ff70, 0xc30560bc, 0xd1c777c8, 0xf1871821, 0x7207ab54, 0x9f5b83c7, 0xb6561c5d,
            0x991e738f, 0xb38b62b9, 0x0ef6d156, 0x994becb1, 0x09a85d0e, 0x32221741, 0xada3cc5f,
            0x5b654ed6, 0x2a7a62b2,
        ],
    },
    Blake3Vector {
        h: [
            0x269e0d37, 0xa6a3a450, 0x892f902b, 0x81e74ef5, 0x099950d8, 0x6f03675a, 0x11e20b8f,
            0x6cad4a26,
        ],
        m: [
            0xf29d0da9, 0x658cda14, 0xf9ebdacc, 0xdbc496cb, 0x4a23d596, 0x2e44158b, 0xa38fd547,
            0x5f557203, 0x34b9b5df, 0x506bf2ef, 0x7403e430, 0x4cbd87ad, 0xcb5c7427, 0x3e7d1bfb,
            0x930d6eaf, 0x86734721,
        ],
        t: 0x12bd4acefaecbd38,
        block_len: 53,
        flags: 42,
        out: [
            0xa632ad45, 0x12ce41f4, 0xd21b2cbd, 0x76795c62, 0x6bec36c1, 0xdafafcde, 0x53ca87b7,
            0x92e8465b, 0x7b424f5d, 0xe1e6ad7f, 0x753ba387, 0xccc50824, 0x69aedf6d, 0xbbbbf253,
            0x78d04883, 0xf3f33689,
        ],
    },
    Blake3Vector {
        h: [
            0x3a096533, 0xf658f7a7, 0x205738d1, 0xb46ee1da, 0x15ceb3a1, 0x359b1548, 0xa4517d6c,
            0x7589ca4a,
        ],
        m: [
            0x74007cb4, 0xd49d0ac1, 0x16edc5d4, 0x685ca8af, 0x4223aa56, 0x10269470, 0x60908405,
            0xa92d04a3, 0x56a3e957, 0xb0f91306, 0xe6c08269, 0xf2306d4a, 0x31a06a7c, 0x9436d6f6,
            0xe18692e2, 0xe0c99f3e,
        ],
        t: 0x329911da9fbd8735,
        block_len: 19,
        flags: 91,
        out: [
            0x913b2ae1, 0xc7f73082, 0x45e1c023, 0x6f1f3f82, 0x20aee6f5, 0xdaf21d94, 0xf2c1e4af,
            0xd4f7d4ac, 0x44a45f87, 0xf4c40ce5, 0x613e9b94, 0x08ce53de, 0x4ff07aa4, 0x456bf2e2,
            0x2066ea7f, 0x3c5a654b,
        ],
    },
    Blake3Vector {
        h: [
            0x5f915ef0, 0x237751aa, 0x01a5ba50, 0x80b65386, 0x14b044d7, 0x61076dc3, 0xb99de255,
            0x283b73a6,
        ],
        m: [
            0x3cee5e2c, 0x1c670ea9, 0x972651da, 0x4a8aa593, 0xac9abb0c, 0x35bb5c11, 0x47fbb3b4,
            0xcf3c17e5, 0xe2eb17c8, 0xe11e99fb, 0x7de0d208, 0x0602fe0c, 0x98cae043, 0x9425b3e2,
            0x33fb4b4f, 0x15607df9,
        ],
        t: 0xeaeb999b8a2e547e,
        block_len: 64,
        flags: 21,
        out: [
            0xf5ee9114, 0x856cabb8, 0x29be2cf1, 0x603be91c, 0x94a7dd0e, 0x28fc3e27, 0xb64e2cc8,
            0x2d2c67ff, 0x69fac1ba, 0x0c949090, 0xd68de435, 0xce91a527, 0xe80c1815, 0x6d44efe6,
            0x87c7b175, 0xd18a8b94,
        ],
    },
];

#[test]
fn test_blake3_6round_canonical_vectors() {
    for (i, v) in CANONICAL_6ROUND_VECTORS.iter().enumerate() {
        let out = blake3_compress_6round(&v.h, &v.m, v.t, v.block_len, v.flags);
        assert_eq!(out, v.out, "canonical 6-round vector {i} mismatch");
    }
}

#[test]
fn test_blake3_syscall_matches_vectors() {
    for (i, v) in CANONICAL_6ROUND_VECTORS.iter().enumerate() {
        let mut pc = 0;
        let mut registers = Registers::default();
        let mut memory = Memory::default();
        let addr = 0x1000u64;

        // Lay out the 176-byte state region: h | m | t | (block_len, flags) | out.
        let mut words = [0u32; 28];
        words[0..8].copy_from_slice(&v.h);
        words[8..24].copy_from_slice(&v.m);
        words[24] = v.t as u32;
        words[25] = (v.t >> 32) as u32;
        words[26] = v.block_len;
        words[27] = v.flags;
        for k in 0..14 {
            let dw = (words[2 * k] as u64) | ((words[2 * k + 1] as u64) << 32);
            memory.store_doubleword(addr + (k as u64) * 8, dw).unwrap();
        }
        // Pre-fill the out region so the test catches a partial write.
        for k in 14..22 {
            memory
                .store_doubleword(addr + (k as u64) * 8, 0xDEAD_BEEF_DEAD_BEEFu64)
                .unwrap();
        }

        registers.write(17, BLAKE3_SYSCALL_NUMBER).unwrap();
        registers.write(10, addr).unwrap();
        Instruction::EcallEbreak
            .run(&mut pc, &mut registers, &mut memory)
            .unwrap();

        let mut got = [0u32; 16];
        for k in 0..8 {
            let dw = memory
                .load_doubleword(addr + ((14 + k) as u64) * 8)
                .unwrap();
            got[2 * k] = dw as u32;
            got[2 * k + 1] = (dw >> 32) as u32;
        }
        assert_eq!(got, v.out, "syscall output mismatch on vector {i}");

        // The 112 input bytes must be untouched.
        for k in 0..14 {
            let dw = memory.load_doubleword(addr + (k as u64) * 8).unwrap();
            let expected = (words[2 * k] as u64) | ((words[2 * k + 1] as u64) << 32);
            assert_eq!(dw, expected, "input dword {k} clobbered on vector {i}");
        }
    }
}

#[test]
fn test_blake3_syscall_rejects_unaligned_state_addr() {
    let mut pc = 0;
    let mut registers = Registers::default();
    let mut memory = Memory::default();

    registers.write(17, BLAKE3_SYSCALL_NUMBER).unwrap();
    registers.write(10, 0x1004).unwrap();

    let err = Instruction::EcallEbreak
        .run(&mut pc, &mut registers, &mut memory)
        .unwrap_err();
    assert!(matches!(
        err,
        ExecutionError::UnalignedBlake3StateAddress(0x1004)
    ));
}

#[test]
fn test_blake3_syscall_rejects_overflowing_state_range() {
    let mut pc = 0;
    let mut registers = Registers::default();
    let mut memory = Memory::default();

    registers.write(17, BLAKE3_SYSCALL_NUMBER).unwrap();
    // 22 dwords = 176 bytes; addr + 175 must not overflow. u64::MAX - 167 is
    // 8-aligned and the last byte lands at u64::MAX + 8 → overflow.
    registers.write(10, u64::MAX - 167).unwrap();

    let err = Instruction::EcallEbreak
        .run(&mut pc, &mut registers, &mut memory)
        .unwrap_err();
    assert!(matches!(
        err,
        ExecutionError::Blake3StateAddressOverflow(addr) if addr == u64::MAX - 167
    ));
}

// =============================================================================
// The chained-absorb ecall's argument validation
// =============================================================================
//
// What the absorb ecall COMPUTES is gated in
// `prover::tables::blake3::executor_absorb_parity`, which drives it against
// `crypto`'s `Blake3Chain` over exhaustive message lengths — this crate cannot,
// having no `crypto` dependency. Checked here is the other half: that every
// argument the chip's constraints assume was validated is in fact rejected when
// it is wrong. A handler that accepted one would hand the prover a trace it
// cannot close, or — worse — one whose memory argument quietly means something
// other than the guest asked for.

/// Registers for a well-formed absorb.
fn absorb_registers(ctrl: u64, msg: u64, num_blocks: u64, first_flags: u64) -> Registers {
    let mut registers = Registers::default();
    registers.write(17, BLAKE3_ABSORB_SYSCALL_NUMBER).unwrap();
    registers.write(10, ctrl).unwrap();
    registers.write(11, msg).unwrap();
    registers.write(12, num_blocks).unwrap();
    registers.write(13, first_flags).unwrap();
    registers
}

fn run_absorb(registers: &mut Registers) -> Result<(), ExecutionError> {
    let mut pc = 0;
    let mut memory = Memory::default();
    Instruction::EcallEbreak
        .run(&mut pc, registers, &mut memory)
        .map(|_| ())
}

#[test]
fn test_blake3_absorb_rejects_unaligned_addresses() {
    // The accelerator reads both regions as doublewords.
    for (ctrl, msg) in [(0x1004u64, 0x2000u64), (0x1000, 0x2004)] {
        let mut registers = absorb_registers(ctrl, msg, 1, 0);
        let err = run_absorb(&mut registers).unwrap_err();
        assert!(
            matches!(err, ExecutionError::UnalignedBlake3AbsorbAddress(_)),
            "ctrl={ctrl:#x} msg={msg:#x} gave {err:?}"
        );
    }
}

#[test]
fn test_blake3_absorb_rejects_block_counts_outside_the_range() {
    // Zero blocks would put an Ecall tuple on the bus with no compression row
    // to answer it.
    let mut registers = absorb_registers(0x1000, 0x2000, 0, 0);
    assert!(matches!(
        run_absorb(&mut registers).unwrap_err(),
        ExecutionError::Blake3AbsorbBlockCountOutOfRange(0)
    ));

    // Past the cap the chip's row counter would no longer be bounded far below
    // the field's modulus, and "the counter reaches 1" would stop implying "the
    // group has exactly `num_blocks` rows".
    let over = BLAKE3_ABSORB_MAX_BLOCKS + 1;
    let mut registers = absorb_registers(0x1000, 0x2000, over, 0);
    assert!(matches!(
        run_absorb(&mut registers).unwrap_err(),
        ExecutionError::Blake3AbsorbBlockCountOutOfRange(n) if n == over
    ));
}

#[test]
fn test_blake3_absorb_rejects_flags_wider_than_the_column() {
    // The chip commits `flags` as one 32-bit word; a wider value would be
    // truncated into the trace and the proof would attest a different hash.
    let mut registers = absorb_registers(0x1000, 0x2000, 1, 1u64 << 32);
    assert!(matches!(
        run_absorb(&mut registers).unwrap_err(),
        ExecutionError::Blake3AbsorbFlagsOutOfRange(f) if f == 1u64 << 32
    ));
}

#[test]
fn test_blake3_absorb_rejects_overlapping_regions() {
    // Every access of one absorb carries the same timestamp, so an address in
    // both regions would be touched twice at that timestamp and the MEMW
    // consistency argument could not order the pair.
    let ctrl = 0x1000u64;
    for msg in [ctrl, ctrl - 64, ctrl + 32, ctrl + 56] {
        let mut registers = absorb_registers(ctrl, msg, 2, 0);
        let err = run_absorb(&mut registers).unwrap_err();
        assert!(
            matches!(err, ExecutionError::Blake3AbsorbRegionOverlap),
            "msg={msg:#x} gave {err:?}"
        );
    }
    // CONTROL: regions that merely abut are accepted — the message's 128 bytes
    // end exactly where the control region starts — so the check is discriminating
    // overlap rather than rejecting every nearby address.
    let mut registers = absorb_registers(ctrl, ctrl - 128, 2, 0);
    assert!(!matches!(
        run_absorb(&mut registers),
        Err(ExecutionError::Blake3AbsorbRegionOverlap)
    ));
}

#[test]
fn test_blake3_absorb_rejects_overflowing_message_range() {
    // `num_blocks * 64` must not wrap and the last byte must be addressable.
    let mut registers = absorb_registers(0x1000, u64::MAX - 63, 2, 0);
    assert!(matches!(
        run_absorb(&mut registers).unwrap_err(),
        ExecutionError::Blake3AbsorbAddressOverflow(_)
    ));
}
