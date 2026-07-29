//! R1a probe adapter: host the production keccak table family in a foreign AIR set.
//!
//! The production keccak family is three chips: `KECCAK` (the core, in
//! `tables::keccak`), `KECCAK_RND` (24 rounds of the permutation, in
//! `tables::keccak_rnd`) and `KECCAK_RC` (the round-constant fixed table). Only
//! the core is VM-coupled — it reads and writes the 25 state lanes through
//! timestamped `MEMW` tokens, so it cannot be lifted into the LFM machine as-is.
//! `KECCAK_RND` and `KECCAK_RC` are pure: they speak only the `Keccak`,
//! `KeccakRc` and BITWISE buses.
//!
//! This module is the minimal chip that replaces the core: it opens the
//! `Keccak` bus with a request token and closes it with the reply token,
//! reproducing exactly the two tokens `tables::keccak::bus_interactions` emits
//! (see `keccak.rs:264-325`) and nothing else. No memory, no timestamps, no
//! address range checks. Feeding those two tokens is the entire contract the
//! round chip needs, so the unchanged `KECCAK_RND` + `KECCAK_RC` + `BITWISE`
//! AIRs prove real `keccak-f[1600]` permutations driven by this adapter.
//!
//! # Token layout
//!
//! Both tokens are 203 bus elements: `[tag_lo, tag_hi, round, state[200]]`,
//! `round = 0` on the request (sent) and `round = 24` on the reply (received).
//! The 200 state elements are traversed **column-major over lanes**: element
//! `3 + 8 * (5x + y) + b` is byte `b` (LSB-first) of lane `x + 5y`, so lanes are
//! visited in the order 0, 5, 10, 15, 20, 1, 6, ... That asymmetry is inherited
//! from the production sender's `for x { for y { for b } } }` loop over
//! `cols::input_state(x, y, b) = INPUT_STATE + (x + 5y) * 8 + b`; emitting the
//! lanes in natural order instead would leave the bus unbalanced.
//!
//! # Constraints
//!
//! The adapter carries no polynomial constraints. Byte-ness of the 400 state
//! columns is enforced transitively rather than locally: every one of the 200
//! IN bytes is an operand of at least one `BYTE_ALU[XOR]` lookup in the round
//! chip's θ column-parity chain (which covers all 25 lanes) and again in its θ
//! final XOR, and every OUT byte is the *result* of a `BYTE_ALU[XOR]` lookup
//! (χ, or ι for lane 0). A non-byte value in any of those columns finds no row
//! in the BITWISE table and breaks the bus balance. The tag columns are pure
//! labels and are deliberately unconstrained here.
//!
//! The production LFM adapter will add what this probe omits: binding the state
//! columns to `LfmMem` words (so the permutation's input and output are the
//! machine's data, not free witness), and sourcing the tag from preprocessed
//! program data.
//!
//! # Tag uniqueness is soundness-critical
//!
//! Nothing but the tag binds a request token to its reply token. Two rows
//! carrying the same tag let a malicious prover swap their output states: the
//! two `(tag, 24, ·)` receives and the two `(tag, 24, ·)` sends still form the
//! same multiset, so the bus balances and the proof verifies. The probe test
//! `duplicate_tag_output_swap_accepts_demonstrating_hazard` pins exactly this.
//! This probe uses distinct per-row constant tags; the production LFM adapter
//! will carry tags as preprocessed program data with registrar-vouched
//! uniqueness, so a prover cannot choose them at all.

use stark::lookup::{BusInteraction, BusValue, Multiplicity, Packing};
use stark::trace::TraceTable;

use crate::tables::bitwise::{BitwiseOperation, BitwiseOperationType};
use crate::tables::keccak_rnd::KeccakRoundOperation;
use crate::tables::types::BusId;
use crate::tables::types::{
    FE, GoldilocksExtension, GoldilocksField, VmTable, dword_wl, zeroed_fe_vec,
};

use super::layout;
use super::word::LfmWord;

type F = GoldilocksField;
type E = GoldilocksExtension;

/// Column layout: `[TAG_LO, TAG_HI, IN[200], OUT[200], MU]`.
pub mod cols {
    /// Low 32 bits of the tag (matches the core chip's `TIMESTAMP_0`, a `DWordWL`).
    pub const TAG_LO: usize = 0;
    /// High 32 bits of the tag (matches the core chip's `TIMESTAMP_1`).
    pub const TAG_HI: usize = 1;
    /// Input state, 200 bytes, lane-major: `IN + (x + 5y) * 8 + b`.
    pub const IN: usize = 2;
    /// Output state, 200 bytes, same indexing as [`IN`].
    pub const OUT: usize = IN + 200; // 202
    /// Is-real column; the multiplicity of both `Keccak` bus tokens.
    pub const MU: usize = OUT + 200; // 402

    pub const NUM_COLUMNS: usize = MU + 1; // 403

    /// Column holding byte `b` of input lane `x + 5y`.
    #[inline]
    pub const fn in_byte(x: usize, y: usize, b: usize) -> usize {
        IN + (x + 5 * y) * 8 + b
    }

    /// Column holding byte `b` of output lane `x + 5y`.
    #[inline]
    pub const fn out_byte(x: usize, y: usize, b: usize) -> usize {
        OUT + (x + 5 * y) * 8 + b
    }
}

/// One permutation: `output = keccak_f1600(input)`, labelled by `tag`.
#[derive(Debug, Clone, Copy)]
pub struct KeccakAdapterOperation {
    /// Binds the request token to its reply token. MUST be unique across rows.
    pub tag: u64,
    pub input: [u64; 25],
}

/// The two `Keccak` bus tokens, mirroring `tables::keccak::bus_interactions`
/// interactions 2 and 3 with the memory-coupled columns dropped.
#[allow(clippy::needless_range_loop)]
pub fn bus_interactions() -> Vec<BusInteraction> {
    let mut interactions = Vec::with_capacity(2);

    let tag_values = || {
        vec![
            BusValue::Packed {
                start_column: cols::TAG_LO,
                packing: Packing::Direct,
            },
            BusValue::Packed {
                start_column: cols::TAG_HI,
                packing: Packing::Direct,
            },
        ]
    };

    // Request: send (tag, 0, input_state[200]).
    {
        let mut values = tag_values();
        values.push(BusValue::constant(0));
        for x in 0..5 {
            for y in 0..5 {
                for b in 0..8 {
                    values.push(BusValue::Packed {
                        start_column: cols::in_byte(x, y, b),
                        packing: Packing::Direct,
                    });
                }
            }
        }
        interactions.push(BusInteraction::sender(
            BusId::Keccak,
            Multiplicity::Column(cols::MU),
            values,
        ));
    }

    // Reply: receive (tag, 24, output_state[200]).
    {
        let mut values = tag_values();
        values.push(BusValue::constant(24));
        for x in 0..5 {
            for y in 0..5 {
                for b in 0..8 {
                    values.push(BusValue::Packed {
                        start_column: cols::out_byte(x, y, b),
                        packing: Packing::Direct,
                    });
                }
            }
        }
        interactions.push(BusInteraction::receiver(
            BusId::Keccak,
            Multiplicity::Column(cols::MU),
            values,
        ));
    }

    interactions
}

/// One row per permutation; padding rows are all-zero (so `MU = 0` and they
/// send nothing).
pub fn generate_adapter_trace(ops: &[KeccakAdapterOperation]) -> TraceTable<F, E> {
    let n_rows = ops.len().next_power_of_two().max(4);
    let mut trace = TraceTable::new_main(
        zeroed_fe_vec(n_rows * cols::NUM_COLUMNS),
        cols::NUM_COLUMNS,
        1,
    );
    let table = &mut trace.main_table;

    for (row, op) in ops.iter().enumerate() {
        let [lo, hi] = dword_wl(op.tag);
        table.set_fe(row, cols::TAG_LO, lo);
        table.set_fe(row, cols::TAG_HI, hi);

        let output = permute(op.input);
        for (lane, (&in_lane, &out_lane)) in op.input.iter().zip(output.iter()).enumerate() {
            table.set_dword_bl(row, cols::IN + lane * 8, in_lane);
            table.set_dword_bl(row, cols::OUT + lane * 8, out_lane);
        }

        table.set_fe(row, cols::MU, FE::one());
    }

    trace
}

/// `keccak-f[1600]` as the VM defines it — the same primitive the production
/// trace builder replays (`trace_builder.rs:635`).
pub fn permute(input: [u64; 25]) -> [u64; 25] {
    let mut state = input;
    executor::vm::instruction::execution::keccak_f1600(&mut state);
    state
}

// =========================================================================
// Machine-word view of the state (the LFM_KECCAK chip's u32-half convention)
// =========================================================================

/// Half `h` of a state: the low (`h` even) or high (`h` odd) 32 bits of lane
/// `h / 2`.
///
/// A keccak lane is a `u64` and so is *not* felt-representable — values in
/// `[p, 2^64)` exist — which is why machine-side keccak state travels as `u32`
/// halves, one per felt lane.
#[inline]
pub fn half_of(state: &[u64; 25], h: usize) -> u32 {
    (state[h / 2] >> (32 * (h % 2))) as u32
}

/// A state as [`layout::keccak::NUM_WORDS`] machine words, four halves each.
///
/// The top `WORD_SLOTS − NUM_HALVES` lanes of the last word are unused and set
/// to zero; the `LFM_KECCAK` chip pins them as bus tuple constants, so a
/// nonzero value there cannot balance.
pub fn state_to_words(state: &[u64; 25]) -> [LfmWord; layout::keccak::NUM_WORDS] {
    core::array::from_fn(|j| {
        core::array::from_fn(|l| {
            let h = 4 * j + l;
            if h < layout::keccak::NUM_HALVES {
                FE::from(u64::from(half_of(state, h)))
            } else {
                FE::zero()
            }
        })
    })
}

/// The `BYTE_ALU[XOR]` lookups the `LFM_KECCAK` chip's absorb rows send: one
/// per rate byte, `PERM_IN[k] = STATE[k] ⊕ BLOCK[k]`.
///
/// Permute rows send none — their XOR interactions are gated by `MODE_ABSORB`.
pub fn absorb_bitwise_ops(rows: &[super::executor::KeccakRow]) -> Vec<BitwiseOperation> {
    use super::instr::KeccakMode;
    let mut out = Vec::new();
    for r in rows.iter().filter(|r| r.mode == KeccakMode::Absorb) {
        for k in 0..layout::keccak::RATE_BYTES {
            let state_byte = (r.state[k / 8] >> (8 * (k % 8))) as u8;
            out.push(BitwiseOperation::byte_op(
                BitwiseOperationType::ByteAluXor,
                state_byte,
                r.block[k],
            ));
        }
    }
    out
}

/// The byte-REVERSED first 32 bytes of a state, as two machine words — the
/// value the production transcript's `sample()` both returns and re-absorbs.
///
/// Reversed byte `j` is digest byte `31 − j`, so both the byte order within a
/// half and the order of the halves flip. The `LFM_KECCAK` chip produces this
/// with two extra bus sends over the SAME output byte columns; this is the host
/// mirror.
pub fn reversed_digest_words(state: &[u64; 25]) -> [LfmWord; 2] {
    let mut digest = [0u8; 32];
    for (lane, chunk) in state[..4].iter().zip(digest.chunks_exact_mut(8)) {
        chunk.copy_from_slice(&lane.to_le_bytes());
    }
    digest.reverse();
    core::array::from_fn(|w| {
        core::array::from_fn(|l| {
            let h = 4 * w + l;
            let mut half = [0u8; 4];
            half.copy_from_slice(&digest[4 * h..4 * h + 4]);
            FE::from(u64::from(u32::from_le_bytes(half)))
        })
    })
}

/// Reassembles the 25 `u64` lanes from 50 `u32` halves.
pub fn halves_to_state(halves: &[u32; layout::keccak::NUM_HALVES]) -> [u64; 25] {
    core::array::from_fn(|lane| {
        u64::from(halves[2 * lane]) | (u64::from(halves[2 * lane + 1]) << 32)
    })
}

/// The `KECCAK_RND` operations matching `ops`: one per permutation, expanding to
/// 24 trace rows each. The round chip keys its rows on `timestamp`, which is our
/// tag. Its `output` field is dead (the trace builder recomputes the state round
/// by round) but is filled with the true value anyway.
pub fn round_operations(ops: &[KeccakAdapterOperation]) -> Vec<KeccakRoundOperation> {
    ops.iter()
        .map(|op| KeccakRoundOperation {
            timestamp: op.tag,
            input: op.input,
            output: permute(op.input),
        })
        .collect()
}

/// BITWISE lookups the `KECCAK_RND` rows of `ops` send: exactly `24 * 1148` per
/// permutation.
///
/// This is the per-round half of `trace_builder::collect_bitwise_from_keccak`,
/// forked rather than called: the original also emits the 105 address-shaped
/// lookups (1 `BYTE_ALU[AND]` alignment check, 4 `ARE_BYTES` on the address
/// bytes, 100 `IS_HALF` on the lane pointers) that belong to the dropped core
/// chip. Calling it with a synthetic address and subtracting would depend on
/// those counts staying fixed; forking the loop keeps the coupling explicit.
#[allow(clippy::needless_range_loop)]
pub fn bitwise_ops_for(ops: &[KeccakAdapterOperation]) -> Vec<BitwiseOperation> {
    use executor::vm::instruction::execution::{KECCAK_RC, KECCAK_RHO};

    let mut out = Vec::with_capacity(ops.len() * 24 * 1148);

    for op in ops {
        let mut state = op.input;
        for round in 0..24 {
            // --- theta: Cxz chain BYTE_ALU[XOR] (160) ---
            let mut cxz = [[[0u8; 8]; 4]; 5];
            for x in 0..5 {
                for b in 0..8 {
                    let v0 = ((state[x] >> (b * 8)) & 0xFF) as u8;
                    let v1 = ((state[x + 5] >> (b * 8)) & 0xFF) as u8;
                    cxz[x][0][b] = v0 ^ v1;
                    out.push(BitwiseOperation::byte_op(
                        BitwiseOperationType::ByteAluXor,
                        v0,
                        v1,
                    ));
                }
                for stage in 1..4usize {
                    let y = stage + 1;
                    for b in 0..8 {
                        let prev = cxz[x][stage - 1][b];
                        let sv = ((state[x + 5 * y] >> (b * 8)) & 0xFF) as u8;
                        cxz[x][stage][b] = prev ^ sv;
                        out.push(BitwiseOperation::byte_op(
                            BitwiseOperationType::ByteAluXor,
                            prev,
                            sv,
                        ));
                    }
                }
            }

            // theta: HWSL for rotated C (20) + ARE_BYTES on Cxz_left (20 pairs).
            // Cxz_right is range-checked by IS_BIT polynomial constraints on the
            // round chip, not via lookups (spec d75944ee).
            let mut rotated_c = [[0u8; 8]; 5];
            for x in 0..5 {
                let c = cxz[x][3];
                for hw in 0..4 {
                    let halfword = (c[hw * 2] as u16) | ((c[hw * 2 + 1] as u16) << 8);
                    let shifted = halfword << 1; // u16 wraps
                    out.push(BitwiseOperation::new(
                        BitwiseOperationType::Hwsl,
                        (halfword & 0xFF) as u8,
                        ((halfword >> 8) & 0xFF) as u8,
                        1,
                    ));
                    out.push(BitwiseOperation::byte_op(
                        BitwiseOperationType::AreBytes,
                        (shifted & 0xFF) as u8,
                        ((shifted >> 8) & 0xFF) as u8,
                    ));
                }
                let mut left_bytes = [0u8; 8];
                let mut right_bits = [0u8; 4];
                for hw in 0..4 {
                    let halfword = (c[hw * 2] as u16) | ((c[hw * 2 + 1] as u16) << 8);
                    let shifted = halfword << 1;
                    left_bytes[hw * 2] = (shifted & 0xFF) as u8;
                    left_bytes[hw * 2 + 1] = ((shifted >> 8) & 0xFF) as u8;
                    right_bits[hw] = (halfword >> 15) as u8;
                }
                for b in 0usize..8 {
                    let right_contribution = if b.is_multiple_of(2) {
                        right_bits[(b / 2 + 3) % 4]
                    } else {
                        0
                    };
                    rotated_c[x][b] = left_bytes[b].wrapping_add(right_contribution);
                }
            }

            // theta: Dxz BYTE_ALU[XOR] (40)
            let mut d_bytes = [[0u8; 8]; 5];
            for x in 0..5 {
                for b in 0..8 {
                    let a = cxz[(x + 4) % 5][3][b];
                    let rb = rotated_c[(x + 1) % 5][b];
                    d_bytes[x][b] = a ^ rb;
                    out.push(BitwiseOperation::byte_op(
                        BitwiseOperationType::ByteAluXor,
                        a,
                        rb,
                    ));
                }
            }

            // theta final: BYTE_ALU[XOR] (200)
            let mut theta_lanes = [0u64; 25];
            for x in 0..5 {
                for y in 0..5 {
                    let lane = state[x + 5 * y];
                    let mut d_lane = 0u64;
                    for b in 0..8 {
                        d_lane |= (d_bytes[x][b] as u64) << (b * 8);
                    }
                    theta_lanes[x + 5 * y] = lane ^ d_lane;
                    for b in 0..8 {
                        let s = ((lane >> (b * 8)) & 0xFF) as u8;
                        out.push(BitwiseOperation::byte_op(
                            BitwiseOperationType::ByteAluXor,
                            s,
                            d_bytes[x][b],
                        ));
                    }
                }
            }

            // rho: HWSL (100) + ARE_BYTES (200 pairs)
            for x in 0..5 {
                for y in 0..5 {
                    let rho_offset = KECCAK_RHO[x][y] as usize;
                    let rnc_val = (rho_offset % 16) as u8;
                    let theta_lane = theta_lanes[x + 5 * y];
                    for hw in 0..4 {
                        let halfword = ((theta_lane >> (hw * 16)) & 0xFFFF) as u16;
                        let (shifted, carry) = if rnc_val == 0 {
                            (halfword, 0u16)
                        } else {
                            (halfword << rnc_val, halfword >> (16 - rnc_val))
                        };
                        out.push(BitwiseOperation::new(
                            BitwiseOperationType::Hwsl,
                            (halfword & 0xFF) as u8,
                            ((halfword >> 8) & 0xFF) as u8,
                            rnc_val,
                        ));
                        out.push(BitwiseOperation::byte_op(
                            BitwiseOperationType::AreBytes,
                            (shifted & 0xFF) as u8,
                            (carry & 0xFF) as u8,
                        ));
                        out.push(BitwiseOperation::byte_op(
                            BitwiseOperationType::AreBytes,
                            ((shifted >> 8) & 0xFF) as u8,
                            ((carry >> 8) & 0xFF) as u8,
                        ));
                    }
                }
            }

            // pi
            let mut pi_lanes = [0u64; 25];
            for x in 0..5 {
                for y in 0..5 {
                    let rotated = theta_lanes[x + 5 * y].rotate_left(KECCAK_RHO[x][y]);
                    let dst_x = y;
                    let dst_y = (2 * x + 3 * y) % 5;
                    pi_lanes[dst_x + 5 * dst_y] = rotated;
                }
            }

            // chi: BYTE_ALU[AND] (200) + BYTE_ALU[XOR] (200)
            let mut chi_lanes = [0u64; 25];
            for x in 0..5 {
                for y in 0..5 {
                    let not_next = !pi_lanes[(x + 1) % 5 + 5 * y];
                    let next2 = pi_lanes[(x + 2) % 5 + 5 * y];
                    let and_val = not_next & next2;
                    chi_lanes[x + 5 * y] = pi_lanes[x + 5 * y] ^ and_val;
                    for b in 0..8 {
                        let not_byte = ((not_next >> (b * 8)) & 0xFF) as u8;
                        let n2_byte = ((next2 >> (b * 8)) & 0xFF) as u8;
                        out.push(BitwiseOperation::byte_op(
                            BitwiseOperationType::ByteAluAnd,
                            not_byte,
                            n2_byte,
                        ));
                        let pi_byte = ((pi_lanes[x + 5 * y] >> (b * 8)) & 0xFF) as u8;
                        let and_byte = ((and_val >> (b * 8)) & 0xFF) as u8;
                        out.push(BitwiseOperation::byte_op(
                            BitwiseOperationType::ByteAluXor,
                            pi_byte,
                            and_byte,
                        ));
                    }
                }
            }

            // iota: BYTE_ALU[XOR] (8)
            let rc_val = KECCAK_RC[round];
            for b in 0..8 {
                let chi_byte = ((chi_lanes[0] >> (b * 8)) & 0xFF) as u8;
                let rc_byte = ((rc_val >> (b * 8)) & 0xFF) as u8;
                out.push(BitwiseOperation::byte_op(
                    BitwiseOperationType::ByteAluXor,
                    chi_byte,
                    rc_byte,
                ));
            }

            chi_lanes[0] ^= rc_val;
            state = chi_lanes;
        }
    }

    out
}
