//! `B3L` — standard 7-round BLAKE3 on the lean tables: what an
//! [`Instr::B3l`](super::instr::Instr::B3l) computes.
//!
//! One instruction is one BLAKE3 compression at counter 0. It reads its
//! chaining value (BLAKE3's IV, or two words of four `u32` lanes) and its
//! 64-byte block (16 `u32` lanes in four words, or 8 field elements in two
//! words, each felt its canonical value's eight big-endian bytes), and writes
//! the 256-bit digest `out[0..8]` as two words of `u32` lanes, which is also
//! the chaining value of a next block. A row may also write the digest
//! byte-reversed as two more words: the transcript's `sample()` returns those
//! bytes and re-absorbs them.
//!
//! The round count is BLAKE3's own seven ([`ROUNDS`]), never the build's
//! `BLAKE3_ROUNDS` (six under the default `blake3-6round` feature), so a
//! compression here is the one the block format's `Blake3Whir` and the
//! `blake3` crate compute.
//!
//! This module holds the instruction's semantics, shared by the executor and
//! the trace fill so the two cannot disagree. The tables that prove it are
//! the lean chip's (`B3L_HEAD`, `B3L_ROUND`, `B3L_XOR`).

use math::field::traits::IsPrimeField;

use crate::tables::types::{FE, GoldilocksField};

use super::blake3::{BLAKE3_IV, blake3_compress_rounds};
use super::layout::b3l as l;
use super::word::LfmWord;

/// Standard BLAKE3's rounds.
pub const ROUNDS: usize = l::ROUNDS;

const _: () = assert!(ROUNDS == crypto::hash::blake3::BLAKE3_STANDARD_ROUNDS);

/// One compression's values: its chaining value and its block. Everything
/// else a row holds (mode, `block_len`, `flags`, addresses, read counts) is
/// the program's.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct B3lValues {
    /// The chaining value: BLAKE3's IV on an IV row.
    pub h: [u32; 8],
    /// The block, in message byte order.
    pub block: [u8; 64],
}

impl B3lValues {
    /// The block's 16 message words, little-endian.
    pub fn message(&self) -> [u32; 16] {
        core::array::from_fn(|i| {
            u32::from_le_bytes([
                self.block[4 * i],
                self.block[4 * i + 1],
                self.block[4 * i + 2],
                self.block[4 * i + 3],
            ])
        })
    }

    /// The compression's 16 output words at counter 0.
    pub fn output(&self, block_len: u8, flags: u8) -> [u32; 16] {
        blake3_compress_rounds(
            &self.h,
            &self.message(),
            0,
            u32::from(block_len),
            u32::from(flags),
            ROUNDS,
        )
    }

    /// The 256-bit digest: the low eight output words.
    pub fn digest(&self, block_len: u8, flags: u8) -> [u32; 8] {
        let out = self.output(block_len, flags);
        core::array::from_fn(|i| out[i])
    }
}

/// BLAKE3's IV as a chaining value.
pub const IV: [u32; 8] = BLAKE3_IV;

/// A digest (or chaining value) as its two memory words of four `u32` lanes.
pub fn digest_words(d: &[u32; 8]) -> [LfmWord; 2] {
    core::array::from_fn(|w| core::array::from_fn(|lane| FE::from(u64::from(d[4 * w + lane]))))
}

/// The extended output, `out[8..16]` of a compression's 16 output words, as
/// two memory words of four `u32` lanes: the second 32 of BLAKE3's 64 output
/// bytes, each word `v[i + 8] ⊕ h[i]` (the chaining value fed forward).
pub fn extended_words(out: &[u32; 16]) -> [LfmWord; 2] {
    core::array::from_fn(|w| {
        core::array::from_fn(|lane| FE::from(u64::from(out[8 + 4 * w + lane])))
    })
}

/// The digest's 32 bytes, little-endian word by word: what `Blake3Std`
/// returns.
pub fn digest_bytes(d: &[u32; 8]) -> [u8; 32] {
    let mut out = [0u8; 32];
    for (i, w) in d.iter().enumerate() {
        out[4 * i..4 * i + 4].copy_from_slice(&w.to_le_bytes());
    }
    out
}

/// The digest's bytes reversed, as two words of four `u32` lanes, each lane
/// four of the reversed bytes little-endian: `LFM_BLAKE3`'s reversed digest,
/// lane for lane.
pub fn reversed_digest_words(d: &[u32; 8]) -> [LfmWord; 2] {
    let mut bytes = digest_bytes(d);
    bytes.reverse();
    core::array::from_fn(|w| {
        core::array::from_fn(|lane| {
            let h = 4 * w + lane;
            FE::from(u64::from(u32::from_le_bytes([
                bytes[4 * h],
                bytes[4 * h + 1],
                bytes[4 * h + 2],
                bytes[4 * h + 3],
            ])))
        })
    })
}

/// A block of 16 `u32` message words, little-endian.
pub fn lane_block(m: &[u32; 16]) -> [u8; 64] {
    let mut out = [0u8; 64];
    for (i, w) in m.iter().enumerate() {
        out[4 * i..4 * i + 4].copy_from_slice(&w.to_le_bytes());
    }
    out
}

/// A block of 8 field elements, each its canonical value's eight big-endian
/// bytes: what `Blake3Whir`'s leaves and `blake3_joint` absorb.
pub fn felt_block(felts: &[FE; 8]) -> [u8; 64] {
    let mut out = [0u8; 64];
    for (f, v) in felts.iter().enumerate() {
        out[8 * f..8 * f + 8].copy_from_slice(&GoldilocksField::canonical(v.value()).to_be_bytes());
    }
    out
}

/// A lane as a `u32`, or `None` at or above `2^32`.
pub fn lane_u32(v: &FE) -> Option<u32> {
    u32::try_from(GoldilocksField::canonical(v.value())).ok()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lfm::builder::{B3Block, B3Cv, Cell, LfmBuilder};
    use crate::lfm::compiler::{LfmProgram, compile};
    use crate::lfm::executor::{LfmExecError, execute};
    use crate::lfm::hash::HasherKind;
    use crypto::hash::blake3::Blake3Std;
    use crypto::hash::blake3::chain::{BLOCK_LEN, block_flags, block_len_of, num_blocks};
    use crypto::merkle_tree::backends::blake3_joint::{joint_leaf, joint_leaf_midstate};
    use digest::Digest;

    /// A deterministic message of `len` bytes.
    fn message(len: usize, seed: u8) -> Vec<u8> {
        (0..len)
            .map(|i| {
                (i as u8)
                    .wrapping_mul(31)
                    .wrapping_add(seed)
                    .wrapping_mul(7)
            })
            .collect()
    }

    /// A deterministic run of felts, some near the top of the field.
    fn felts(n: usize, seed: u64) -> Vec<FE> {
        (0..n as u64)
            .map(|i| {
                if i % 5 == 3 {
                    // p − 1 − i: canonical values whose high half is maximal.
                    FE::from(0xFFFF_FFFF_0000_0000u64 - i)
                } else {
                    FE::from(
                        i.wrapping_add(seed)
                            .wrapping_mul(0x9E37_79B9_7F4A_7C15)
                            .wrapping_add(7),
                    )
                }
            })
            .collect()
    }

    /// The memory words a run of `u32`s packs into, four lanes a word.
    fn lane_words(b: &mut LfmBuilder, words: &[u32]) -> Vec<Cell> {
        words
            .chunks(4)
            .map(|c| {
                let lanes: LfmWord =
                    core::array::from_fn(|k| FE::from(u64::from(c.get(k).copied().unwrap_or(0))));
                b.digest_const(lanes).as_cell()
            })
            .collect()
    }

    /// Emit `Blake3Std(bytes)` as the machine chains it: block 0 from the IV,
    /// counter 0, `chain::block_flags` / `block_len_of`, every block as lanes.
    /// Returns the digest's cells.
    fn emit_std(b: &mut LfmBuilder, bytes: &[u8]) -> [Cell; 2] {
        let n = num_blocks(bytes.len());
        let mut cv = B3Cv::Iv;
        let mut digest = None;
        for i in 0..n {
            let mut block = [0u8; BLOCK_LEN];
            let start = (i * BLOCK_LEN).min(bytes.len());
            let end = ((i + 1) * BLOCK_LEN).min(bytes.len());
            block[..end - start].copy_from_slice(&bytes[start..end]);
            let m: [u32; 16] = core::array::from_fn(|k| {
                u32::from_le_bytes([
                    block[4 * k],
                    block[4 * k + 1],
                    block[4 * k + 2],
                    block[4 * k + 3],
                ])
            });
            let words = lane_words(b, &m);
            let out = b.b3_compress(
                cv,
                B3Block::Lanes([words[0], words[1], words[2], words[3]]),
                block_len_of(i, bytes.len()) as u8,
                block_flags(i, n) as u8,
            );
            cv = B3Cv::Words(out);
            digest = Some(out);
        }
        digest.expect("at least one block")
    }

    fn run(program: &LfmProgram) -> crate::lfm::executor::LfmExecution {
        execute(program, &[], &HasherKind::Test).expect("the program executes")
    }

    /// The digest the machine wrote at `cells`, as `Blake3Std`'s bytes.
    fn read_digest(exec: &crate::lfm::executor::LfmExecution, cells: [Cell; 2]) -> [u8; 32] {
        let mut out = [0u8; 32];
        for (w, cell) in cells.iter().enumerate() {
            let word = exec.memory.get(cell.addr()).expect("written");
            for (lane, v) in word.iter().enumerate() {
                let x = lane_u32(v).expect("a u32 lane");
                out[16 * w + 4 * lane..16 * w + 4 * lane + 4].copy_from_slice(&x.to_le_bytes());
            }
        }
        out
    }

    /// ★ KAT: up to one chunk the machine's chain is the `blake3` crate's
    /// hash, and at every length (past one chunk too) it is `Blake3Std`.
    #[test]
    fn the_machine_chain_is_blake3_and_blake3std() {
        for &len in &[0usize, 1, 40, 63, 64, 65, 128, 1000, 1024, 1088, 1600] {
            let bytes = message(len, len as u8);
            let mut b = LfmBuilder::new();
            let cells = emit_std(&mut b, &bytes);
            let program = compile(b.finish());
            crate::lfm::validator::validate(&program).expect("admissible");
            let exec = run(&program);
            let got = read_digest(&exec, cells);
            let std: [u8; 32] = <Blake3Std as Digest>::digest(&bytes).into();
            assert_eq!(got, std, "Blake3Std at {len} bytes");
            if len <= 1024 {
                assert_eq!(
                    got,
                    *blake3::hash(&bytes).as_bytes(),
                    "blake3 at {len} bytes"
                );
            }
            assert_eq!(program.groups.b3l.real_rows, num_blocks(len));
        }
    }

    /// ★ KAT: a joint leaf from its midstate, the values read as felt words,
    /// is `blake3_joint::joint_leaf` — the base's round-0 leaf under 7-round
    /// BLAKE3, the in-guest verifier's replay of it.
    #[test]
    fn a_joint_leaf_from_its_midstate_is_the_bases() {
        for &(blocks, per_block) in &[(2usize, 64usize), (3, 64), (4, 64), (3, 8)] {
            let values = felts(blocks * per_block, blocks as u64);
            let expected = joint_leaf(&values, blocks).expect("a joint leaf");
            let mut b = LfmBuilder::new();
            let mid = joint_leaf_midstate(blocks as u64, per_block as u64, 8);
            let mid_cells = lane_words(&mut b, &mid);
            let mut cv = B3Cv::Words([mid_cells[0], mid_cells[1]]);
            let len = crypto::merkle_tree::backends::blake3_joint::JOINT_LEAF_PREFIX_LEN
                + 8 * values.len();
            let n = num_blocks(len);
            let mut digest = None;
            for (i, chunk) in values.chunks(8).enumerate() {
                let words: Vec<Cell> = chunk
                    .chunks(4)
                    .map(|c| {
                        let w: LfmWord = core::array::from_fn(|k| c[k]);
                        b.digest_const(w).as_cell()
                    })
                    .collect();
                let index = i + 1;
                let out = b.b3_compress(
                    cv,
                    B3Block::Felts([words[0], words[1]]),
                    block_len_of(index, len) as u8,
                    block_flags(index, n) as u8,
                );
                cv = B3Cv::Words(out);
                digest = Some(out);
            }
            let program = compile(b.finish());
            crate::lfm::validator::validate(&program).expect("admissible");
            let exec = run(&program);
            assert_eq!(
                read_digest(&exec, digest.expect("a block")),
                expected,
                "joint leaf of {blocks} × {per_block}"
            );
        }
    }

    /// ★ KAT: a Merkle parent — one compression of `left ‖ right` from the IV,
    /// `CHUNK_START | CHUNK_END | ROOT` — is `Blake3Std` over the 64 bytes.
    #[test]
    fn a_parent_is_blake3_over_left_and_right() {
        let left: [u8; 32] = blake3::hash(b"left").into();
        let right: [u8; 32] = blake3::hash(b"right").into();
        let mut bytes = [0u8; 64];
        bytes[..32].copy_from_slice(&left);
        bytes[32..].copy_from_slice(&right);
        let mut b = LfmBuilder::new();
        let cells = emit_std(&mut b, &bytes);
        let program = compile(b.finish());
        let exec = run(&program);
        assert_eq!(read_digest(&exec, cells), *blake3::hash(&bytes).as_bytes());
        let flags = crypto::hash::blake3::chain::FLAGS_ONE_BLOCK as u8;
        let op = program
            .instrs
            .iter()
            .find_map(|i| match i {
                crate::lfm::instr::Instr::B3l(op) => Some(op),
                _ => None,
            })
            .expect("one compression");
        assert_eq!((op.block_len, op.flags), (64, flags));
    }

    /// The reversed digest is the digest's bytes back to front, lane for lane
    /// `LFM_BLAKE3`'s.
    #[test]
    fn the_reversed_digest_is_the_digest_backwards() {
        let bytes = message(64, 9);
        let m: [u32; 16] = core::array::from_fn(|k| {
            u32::from_le_bytes([
                bytes[4 * k],
                bytes[4 * k + 1],
                bytes[4 * k + 2],
                bytes[4 * k + 3],
            ])
        });
        let mut b = LfmBuilder::new();
        let words = lane_words(&mut b, &m);
        let flags = crypto::hash::blake3::chain::FLAGS_ONE_BLOCK as u8;
        let (digest, rev) = b.b3_compress_rev(
            B3Cv::Iv,
            B3Block::Lanes([words[0], words[1], words[2], words[3]]),
            64,
            flags,
        );
        let program = compile(b.finish());
        crate::lfm::validator::validate(&program).expect("admissible");
        let exec = run(&program);
        let mut plain = read_digest(&exec, digest);
        assert_eq!(plain, *blake3::hash(&bytes).as_bytes());
        plain.reverse();
        assert_eq!(read_digest(&exec, rev), plain);
    }

    /// A lane word with a lane at or above 2^32 has no witness: the executor
    /// refuses it rather than produce an unprovable trace.
    #[test]
    fn a_lane_at_or_above_two_to_the_32_is_refused() {
        let mut b = LfmBuilder::new();
        let ok = b.digest_const([FE::from(1u64); 4]).as_cell();
        let bad = b
            .digest_const([FE::from(1u64), FE::from(1u64 << 32), FE::zero(), FE::zero()])
            .as_cell();
        b.b3_compress(B3Cv::Iv, B3Block::Lanes([ok, bad, ok, ok]), 64, 11);
        let program = compile(b.finish());
        let err = execute(&program, &[], &HasherKind::Test).expect_err("refused");
        assert!(
            matches!(err, LfmExecError::NotU32Half { lane: 1, .. }),
            "{err:?}"
        );
        let mut b = LfmBuilder::new();
        let ok = b.digest_const([FE::from(1u64); 4]).as_cell();
        let bad = bad_cell(&mut b);
        b.b3_compress(B3Cv::Words([ok, bad]), B3Block::Lanes([ok; 4]), 64, 11);
        let program = compile(b.finish());
        assert!(matches!(
            execute(&program, &[], &HasherKind::Test),
            Err(LfmExecError::NotU32Half { lane: 3, .. })
        ));
    }

    fn bad_cell(b: &mut LfmBuilder) -> Cell {
        b.digest_const([FE::zero(), FE::zero(), FE::zero(), FE::from(u64::MAX >> 1)])
            .as_cell()
    }

    /// A felt block holds the felts' canonical big-endian bytes, the record
    /// carries them, and the digest is `Blake3Std` over those bytes.
    #[test]
    fn a_felt_block_is_the_felts_big_endian_bytes() {
        let values = felts(8, 3);
        let mut b = LfmBuilder::new();
        let w0 = b
            .digest_const(core::array::from_fn(|k| values[k]))
            .as_cell();
        let w1 = b
            .digest_const(core::array::from_fn(|k| values[4 + k]))
            .as_cell();
        let flags = crypto::hash::blake3::chain::FLAGS_ONE_BLOCK as u8;
        let out = b.b3_compress(B3Cv::Iv, B3Block::Felts([w0, w1]), 64, flags);
        let program = compile(b.finish());
        let exec = run(&program);
        let bytes = felt_block(&core::array::from_fn(|k| values[k]));
        assert_eq!(read_digest(&exec, out), *blake3::hash(&bytes).as_bytes());
        assert_eq!(exec.records.b3l.len(), 1);
        assert_eq!(exec.records.b3l[0].block, bytes);
        assert_eq!(exec.records.b3l[0].h, IV);
    }

    /// One small program with a compression whose digest is published.
    fn one_compression() -> LfmProgram {
        let mut b = LfmBuilder::new();
        let w: Vec<Cell> = (0..4u64)
            .map(|i| b.digest_const([FE::from(i); 4]).as_cell())
            .collect();
        let flags = crypto::hash::blake3::chain::FLAGS_ONE_BLOCK as u8;
        let d = b.b3_compress(
            B3Cv::Iv,
            B3Block::Lanes([w[0], w[1], w[2], w[3]]),
            64,
            flags,
        );
        b.public(d[0]);
        b.public(d[1]);
        compile(b.finish())
    }

    /// ⛔ The STARK chip set has no lean tables: its prover refuses a program
    /// that carries a compression, before anything reaches the card.
    #[test]
    fn the_stark_prover_refuses_a_lean_compression() {
        let program = one_compression();
        crate::lfm::validator::validate(&program).expect("admissible");
        let opts =
            stark::proof::options::GoldilocksCubicProofOptions::with_blowup(2).expect("options");
        let artifacts = crate::lfm::registry::build_artifacts(&program, &opts);
        let Err(err) = crate::lfm::proof::lfm_prove(&program, &artifacts, &[], &opts) else {
            panic!("the STARK prover proved a lean compression");
        };
        assert!(format!("{err:?}").contains("lean BLAKE3"), "{err:?}");
    }

    /// The validator re-checks the head group: a tag off its ordinal, a
    /// `block_len` past 64, a felt block with a third word, an IV row with a
    /// chaining-value word — each refused.
    #[test]
    fn the_validator_refuses_a_malformed_head_row() {
        use crate::lfm::validator::{LfmViolation, validate};
        let tamper = |col: usize, v: u64| {
            let mut program = one_compression();
            program.groups.b3l.set(0, col, FE::from(v));
            validate(&program)
        };
        for (col, v) in [
            (l::TAG, 2),
            (l::BLEN, 65),
            (l::FLAGS, 256),
            (l::haddr(1), 5),
        ] {
            assert!(
                matches!(
                    tamper(col, v),
                    Err(LfmViolation::MalformedB3lRow { row: 0 })
                ),
                "column {col} = {v}"
            );
        }
        // A felt block whose spare word slots are not zero.
        let mut b = LfmBuilder::new();
        let w = b.digest_const([FE::from(3u64); 4]).as_cell();
        b.b3_compress(B3Cv::Iv, B3Block::Felts([w, w]), 64, 11);
        let mut program = compile(b.finish());
        validate(&program).expect("admissible");
        program.groups.b3l.set(0, l::maddr(3), FE::from(1u64));
        assert!(matches!(
            validate(&program),
            Err(LfmViolation::MalformedB3lRow { row: 0 })
        ));
        // Both block modes, or neither.
        let mut program = one_compression();
        program.groups.b3l.set(0, l::MFELTS, FE::one());
        assert!(matches!(
            validate(&program),
            Err(LfmViolation::NonOneHotSelector {
                chip: "B3L_HEAD",
                row: 0
            })
        ));
    }

    /// The head group: one row a compression, its mode, constants, addresses,
    /// read counts and tag laid out as `layout::b3l` says, padding zero.
    #[test]
    fn the_head_group_is_laid_out_as_the_layout_says() {
        let mut b = LfmBuilder::new();
        let w: Vec<Cell> = (0..4u64)
            .map(|i| b.digest_const([FE::from(i); 4]).as_cell())
            .collect();
        let d0 = b.b3_compress(B3Cv::Iv, B3Block::Lanes([w[0], w[1], w[2], w[3]]), 64, 1);
        let (d1, r1) = b.b3_compress_rev(B3Cv::Words(d0), B3Block::Felts([w[0], w[1]]), 17, 10);
        for c in d1.iter().chain(&r1).chain(&r1) {
            b.public(*c);
        }
        let program = compile(b.finish());
        crate::lfm::validator::validate(&program).expect("admissible");
        let g = &program.groups.b3l;
        assert_eq!((g.width, g.real_rows), (l::PREP_WIDTH, 2));
        let at = |r: usize, c: usize| GoldilocksField::canonical(g.at(r, c).value());
        assert_eq!(
            (
                at(0, l::MU),
                at(0, l::TAG),
                at(0, l::IVF),
                at(0, l::MLANES),
                at(0, l::MFELTS)
            ),
            (1, 1, 1, 1, 0),
            "row 0: real, tag 1, from the IV, a lane block"
        );
        assert_eq!((at(1, l::TAG), at(1, l::IVF), at(1, l::MFELTS)), (2, 0, 1));
        assert_eq!(
            (
                at(0, l::BLEN),
                at(0, l::FLAGS),
                at(1, l::BLEN),
                at(1, l::FLAGS)
            ),
            (64, 1, 17, 10)
        );
        // Row 0's digest is read twice, as row 1's chaining value.
        assert_eq!((at(0, l::ocnt(0)), at(0, l::ocnt(1))), (1, 1));
        assert_eq!(at(1, l::haddr(0)), d0[0].addr().0);
        assert_eq!(
            at(1, l::maddr(2)),
            0,
            "a felt block's spare word slots are 0"
        );
        // Row 1's digest is public once; its reversed words twice.
        assert_eq!(
            (at(1, l::ocnt(0)), at(1, l::rcnt(0)), at(1, l::rcnt(1))),
            (1, 2, 2)
        );
        assert_eq!(at(1, l::raddr(1)), r1[1].addr().0);
        for r in 2..g.padded_rows {
            for c in 0..g.width {
                assert_eq!(at(r, c), 0, "padding row {r} column {c}");
            }
        }
    }

    /// ★ KAT: a compression asked for its extended output writes, past its
    /// digest, the second 32 of the `blake3` crate's 64 bytes of extended
    /// output — on a one-block message (chaining value the IV) and past the
    /// first block (a previous block's digest fed forward). Its group is the
    /// wider one, with the extended words' addresses and read counts on that
    /// row and zero on every other.
    #[test]
    fn the_extended_output_is_the_blake3_crates_xof() {
        for &len in &[0usize, 40, 64, 65, 200, 1024] {
            let bytes = message(len, 3 + len as u8);
            let n = num_blocks(len);
            let mut b = LfmBuilder::new();
            let mut cv = B3Cv::Iv;
            let mut last = None;
            for i in 0..n {
                let mut block = [0u8; BLOCK_LEN];
                let start = (i * BLOCK_LEN).min(len);
                let end = ((i + 1) * BLOCK_LEN).min(len);
                block[..end - start].copy_from_slice(&bytes[start..end]);
                let m: [u32; 16] = core::array::from_fn(|k| {
                    u32::from_le_bytes(block[4 * k..4 * k + 4].try_into().expect("4 bytes"))
                });
                let words = lane_words(&mut b, &m);
                let block = B3Block::Lanes([words[0], words[1], words[2], words[3]]);
                let (blen, flags) = (block_len_of(i, len) as u8, block_flags(i, n) as u8);
                if i + 1 == n {
                    last = Some(b.b3_compress_rev_ext(cv, block, blen, flags));
                } else {
                    cv = B3Cv::Words(b.b3_compress(cv, block, blen, flags));
                }
            }
            let (digest, _rev, ext) = last.expect("a last block");
            for c in digest.iter().chain(&ext).chain(&ext) {
                b.public(*c);
            }
            let program = compile(b.finish());
            crate::lfm::validator::validate(&program).expect("admissible");
            let exec = run(&program);
            let mut want = [0u8; 64];
            let mut hasher = blake3::Hasher::new();
            hasher.update(&bytes);
            hasher.finalize_xof().fill(&mut want);
            assert_eq!(read_digest(&exec, digest), want[..32], "digest at {len}");
            assert_eq!(
                read_digest(&exec, ext),
                want[32..],
                "extended output at {len}"
            );
            let g = &program.groups.b3l;
            assert_eq!((g.width, g.real_rows), (l::PREP_WIDTH_XOF, n));
            let at = |r: usize, c: usize| GoldilocksField::canonical(g.at(r, c).value());
            for r in 0..g.padded_rows {
                let is_last = r + 1 == n;
                for (j, word) in ext.iter().enumerate() {
                    let (addr, cnt) = (at(r, l::xaddr(j)), at(r, l::xcnt(j)));
                    if is_last {
                        assert_eq!((addr, cnt), (word.addr().0, 2), "row {r} word {j}");
                    } else {
                        assert_eq!((addr, cnt), (0, 0), "row {r} word {j}");
                    }
                }
            }
        }
    }

    /// A program with no extended output keeps today's group width.
    #[test]
    fn no_extended_output_keeps_the_narrow_group() {
        let mut b = LfmBuilder::new();
        let d = emit_std(&mut b, &message(100, 9));
        for c in d {
            b.public(c);
        }
        let program = compile(b.finish());
        assert_eq!(program.groups.b3l.width, l::PREP_WIDTH);
    }
}
