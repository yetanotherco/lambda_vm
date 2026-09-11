//! KECCAK_RND: Round chip for Keccak-f[1600] permutation.
//!
//! One row per round (24 rows per keccak call). Bitwise XOR/AND are delegated
//! to BITWISE lookup tables (BYTE_ALU, ARE_BYTES); the halfword shifts (θ
//! rotate-by-1 and ρ) are enforced directly by μ-gated linear identities over
//! the committed shift cells instead of HWSL lookups (see
//! `KeccakRndConstraints`). ARE_BYTES range checks on the shift outputs and the
//! IS_BIT constraint on the θ carry are load-bearing for the identities.
//!
//! ## Column layout (1,480 columns)
//!
//! | Group          | Size | Description                                       |
//! |----------------|------|---------------------------------------------------|
//! | timestamp      |    2 | DWordWL                                           |
//! | round          |    1 | Round index (0..23)                               |
//! | start          |  200 | Input state bytes [5][5][8]                       |
//! | Cxz            |  160 | Column parity chain [5][4][8]                     |
//! | Cxz_left       |   40 | Left component of rotated C [5][8]                |
//! | Cxz_right      |   20 | Carry bits of HWSL(C[x],1) [5][4]                 |
//! | Dxz            |   40 | D values [5][8]                                   |
//! | theta          |  200 | State after θ [5][5][8]                           |
//! | rot_left       |  200 | Left half of ρ rotation [5][5][8]                 |
//! | rot_right      |  200 | Right half of ρ rotation [5][5][8]                |
//! | chi_ands       |  200 | AND results for χ [5][5][8]                       |
//! | chi            |  200 | State after χ [5][5][8]                           |
//! | rc             |    8 | Round constant bytes                              |
//! | iota           |    8 | χ[0][0] ⊕ rc                                      |
//! | mu             |    1 | Multiplicity (1 for real, 0 for padding)          |
//!
//! Note: spec [[variables.constant]] `rnc` and `rbc` are inlined as compile-time
//! constants derived from `KECCAK_RHO[x][y]`, not materialized as columns.
//! `Cxz_right` is typed `[Bit, 4]` per spec d75944ee — a halfword rotate-by-1
//! carries out a single bit, range-checked via IS_BIT polynomial constraints.

use executor::vm::instruction::execution::{KECCAK_RC, KECCAK_RHO};
use stark::constraints::builder::{ConstraintBuilder, ConstraintSet};
use stark::lookup::{BusInteraction, BusValue, LinearTerm, Multiplicity, Packing};
use stark::trace::TraceTable;

use super::types::{BusId, FE, GoldilocksExtension, GoldilocksField, VmTable, alu_op};

// =========================================================================
// Column indices
// =========================================================================

pub mod cols {
    pub const TIMESTAMP_0: usize = 0;
    pub const TIMESTAMP_1: usize = 1;
    pub const ROUND: usize = 2;

    // start[5][5][8] = 200 bytes — input state for this round
    pub const START: usize = 3;

    // Cxz[5][4][8] = 160 bytes — partial XOR chain for column parities
    pub const CXZ: usize = START + 200; // 203

    // Cxz_left[5][8] = 40 bytes — left shift component of rotated C
    pub const CXZ_LEFT: usize = CXZ + 160; // 363

    // Cxz_right[5][4] = 20 bits — carry bit of HWSL(C[x] halfword[hw], 1).
    // For shift=1, HWSL emits a single-bit carry; one column per halfword.
    pub const CXZ_RIGHT: usize = CXZ_LEFT + 40; // 403

    // Dxz[5][8] = 40 bytes
    pub const DXZ: usize = CXZ_RIGHT + 20; // 423

    // theta[5][5][8] = 200 bytes — state after θ
    pub const THETA: usize = DXZ + 40; // 463

    // rot_left[5][5][8] = 200 bytes
    pub const ROT_LEFT: usize = THETA + 200; // 663

    // rot_right[5][5][8] = 200 bytes
    pub const ROT_RIGHT: usize = ROT_LEFT + 200; // 863

    // chi_ands[5][5][8] = 200 bytes
    // (pi is a spec [[variables.virtual]] — inlined as rot_left + rot_right at
    // compile-resolved offsets, not materialized as columns.)
    pub const CHI_ANDS: usize = ROT_RIGHT + 200; // 1063

    // chi[5][5][8] = 200 bytes — state after χ
    pub const CHI: usize = CHI_ANDS + 200; // 1263

    // rc[8] — round constant bytes
    pub const RC: usize = CHI + 200; // 1463

    // iota[8] — χ[0][0] ⊕ rc
    pub const IOTA: usize = RC + 8; // 1471

    // mu — multiplicity flag.
    // rnc and rbc (spec [[variables.constant]]) are inlined as compile-time
    // constants from KECCAK_RHO, not allocated as columns.
    pub const MU: usize = IOTA + 8; // 1479

    pub const NUM_COLUMNS: usize = MU + 1; // 1480

    // -------------------------------------------------------------------------
    // Index helpers
    // -------------------------------------------------------------------------

    /// Index into start[x][y][byte] (200 bytes, row-major: y varies fastest)
    #[inline]
    pub const fn start(x: usize, y: usize, byte: usize) -> usize {
        START + (x + 5 * y) * 8 + byte
    }

    /// Index into Cxz[x][stage][byte] (160 bytes)
    #[inline]
    pub const fn cxz(x: usize, stage: usize, byte: usize) -> usize {
        CXZ + (x * 4 + stage) * 8 + byte
    }

    /// Index into Cxz_left[x][byte]
    #[inline]
    pub const fn cxz_left(x: usize, byte: usize) -> usize {
        CXZ_LEFT + x * 8 + byte
    }

    /// Index into Cxz_right[x][hw] — single-bit carry for halfword `hw` of x.
    #[inline]
    pub const fn cxz_right_bit(x: usize, hw: usize) -> usize {
        CXZ_RIGHT + x * 4 + hw
    }

    /// For byte `b` of the rotated_Cxz output, return Some(hw) if a Cxz_right
    /// bit contributes (even b), else None (odd b → only Cxz_left contributes).
    /// Spec d75944ee/9143370f: rotated_Cxz[z] = Cxz_left[z] + (1 - z%2) *
    /// Cxz_right[(z/2 - 1) mod 4].
    #[inline]
    pub const fn cxz_right_bit_for_byte(b: usize) -> Option<usize> {
        if b.is_multiple_of(2) {
            Some((b / 2 + 3) % 4)
        } else {
            None
        }
    }

    /// Index into Dxz[x][byte]
    #[inline]
    pub const fn dxz(x: usize, byte: usize) -> usize {
        DXZ + x * 8 + byte
    }

    /// Index into theta[x][y][byte]
    #[inline]
    pub const fn theta(x: usize, y: usize, byte: usize) -> usize {
        THETA + (x + 5 * y) * 8 + byte
    }

    /// Index into rot_left[x][y][byte]
    #[inline]
    pub const fn rot_left(x: usize, y: usize, byte: usize) -> usize {
        ROT_LEFT + (x + 5 * y) * 8 + byte
    }

    /// Index into rot_right[x][y][byte]
    #[inline]
    pub const fn rot_right(x: usize, y: usize, byte: usize) -> usize {
        ROT_RIGHT + (x + 5 * y) * 8 + byte
    }

    /// Resolve pi[x][y][z] (spec virtual) to the (rot_left_col, rot_right_col)
    /// pair whose sum equals pi[x][y][z]. rbc is compile-time constant.
    #[inline]
    pub fn pi_src_cols(x: usize, y: usize, z: usize) -> (usize, usize) {
        use executor::vm::instruction::execution::KECCAK_RHO;
        let sx = (x + 3 * y) % 5;
        let sy = x;
        let rho_offset = KECCAK_RHO[sx][sy] as usize;
        let rbc_val = rho_offset / 16;
        let (l_byte, r_byte) = match rbc_val {
            0 => (z, (z + 6) % 8),
            1 => ((z + 6) % 8, (z + 4) % 8),
            2 => ((z + 4) % 8, (z + 2) % 8),
            3 => ((z + 2) % 8, z),
            _ => unreachable!(),
        };
        (rot_left(sx, sy, l_byte), rot_right(sx, sy, r_byte))
    }

    /// Index into chi_ands[x][y][byte]
    #[inline]
    pub const fn chi_ands(x: usize, y: usize, byte: usize) -> usize {
        CHI_ANDS + (x + 5 * y) * 8 + byte
    }

    /// Index into chi[x][y][byte]
    #[inline]
    pub const fn chi(x: usize, y: usize, byte: usize) -> usize {
        CHI + (x + 5 * y) * 8 + byte
    }

    /// Index into rc[byte]
    #[inline]
    pub const fn rc(byte: usize) -> usize {
        RC + byte
    }

    /// Index into iota[byte]
    #[inline]
    pub const fn iota(byte: usize) -> usize {
        IOTA + byte
    }
}

// =========================================================================
// Operation struct
// =========================================================================

/// Trace rows one [`KeccakRoundOperation`] expands into, one per keccak round.
/// Chunking splits on whole operations, so a chunk limit in rows divides by this.
pub const ROUNDS_PER_OP: usize = 24;

/// One keccak permutation call's worth of data (produces 24 rows).
#[derive(Debug, Clone)]
pub struct KeccakRoundOperation {
    pub timestamp: u64,
    pub input: [u64; 25],
    pub output: [u64; 25],
}

// =========================================================================
// Trace generation
// =========================================================================

/// Extract byte `b` (0..8) from a u64 value.
#[inline]
fn byte_of(val: u64, b: usize) -> u8 {
    ((val >> (b * 8)) & 0xFF) as u8
}

/// Compute halfword shift left: (value << shift) mod 2^16 and value >> (16 - shift).
#[inline]
fn hwsl(halfword: u16, shift: u8) -> (u16, u16) {
    if shift == 0 {
        (halfword, 0)
    } else {
        (
            halfword << shift, // u16 naturally wraps at 16 bits
            halfword >> (16 - shift),
        )
    }
}

#[allow(clippy::needless_range_loop)]
/// Generate the KECCAK_RND trace table.
///
/// Each `KeccakRoundOperation` produces 24 rows (one per round). The trace
/// computes all intermediate values (θ, ρ, π, χ, ι) at byte granularity.
pub fn generate_keccak_rnd_trace(
    ops: &[KeccakRoundOperation],
) -> TraceTable<GoldilocksField, GoldilocksExtension> {
    let n_rows = (ops.len() * ROUNDS_PER_OP).next_power_of_two().max(4);
    let mut trace = TraceTable::new_main(
        crate::tables::types::zeroed_fe_vec(n_rows * cols::NUM_COLUMNS),
        cols::NUM_COLUMNS,
        1,
    );
    let table = &mut trace.main_table;

    for (op_idx, op) in ops.iter().enumerate() {
        // Execute round-by-round, tracking the state
        let mut state = op.input;

        for round in 0..24 {
            let row_idx = op_idx * 24 + round;

            // Timestamp & round
            table.set_dword_wl(row_idx, cols::TIMESTAMP_0, op.timestamp);
            table.set_u64(row_idx, cols::ROUND, round as u64);

            // start = current state as bytes
            for x in 0..5 {
                for y in 0..5 {
                    let lane = state[x + 5 * y];
                    table.set_dword_bl(row_idx, cols::start(x, y, 0), lane);
                }
            }

            // === θ (theta) ===
            // Column parities: C[x] = XOR of all 5 lanes in column x
            // Computed as a chain: Cxz[x][0] = start[x,0] XOR start[x,1]
            //                      Cxz[x][k] = Cxz[x][k-1] XOR start[x,k+1]
            let mut c_bytes = [[0u8; 8]; 5]; // C[x][byte] = final parity
            let mut cxz = [[[0u8; 8]; 4]; 5]; // Cxz[x][stage][byte]
            for x in 0..5 {
                // Stage 0: XOR(start[x,0], start[x,1])
                for b in 0..8 {
                    let v0 = byte_of(state[x], b);
                    let v1 = byte_of(state[x + 5], b);
                    cxz[x][0][b] = v0 ^ v1;
                }
                table.set_bytes(row_idx, cols::cxz(x, 0, 0), &cxz[x][0]);

                // Stages 1..3: XOR(Cxz[x][k-1], start[x, k+1])
                for stage in 1..4 {
                    let y = stage + 1;
                    for b in 0..8 {
                        let prev = cxz[x][stage - 1][b];
                        let sv = byte_of(state[x + 5 * y], b);
                        cxz[x][stage][b] = prev ^ sv;
                    }
                    table.set_bytes(row_idx, cols::cxz(x, stage, 0), &cxz[x][stage]);
                }
                c_bytes[x] = cxz[x][3];
            }

            // Rotate C left by 1 bit using HWSL decomposition.
            // HWSL shifts each halfword (u16) independently. For shift=1, the
            // carry is a single bit (top bit of the halfword); we store it in
            // one column per halfword (Cxz_right[x][hw], spec d75944ee).
            //   rotated_Cxz[z] = Cxz_left[z] + (1 - z%2) * Cxz_right[(z/2 - 1) mod 4]
            let mut cxz_left_bytes = [[0u8; 8]; 5];
            let mut cxz_right_bits = [[0u8; 4]; 5];
            let mut rotated_c = [[0u8; 8]; 5];
            for x in 0..5 {
                for hw in 0..4 {
                    let lo = c_bytes[x][hw * 2] as u16;
                    let hi = c_bytes[x][hw * 2 + 1] as u16;
                    let halfword = lo | (hi << 8);
                    let (shifted, carry) = hwsl(halfword, 1);
                    cxz_left_bytes[x][hw * 2] = (shifted & 0xFF) as u8;
                    cxz_left_bytes[x][hw * 2 + 1] = (shifted >> 8) as u8;
                    // For shift=1, carry ∈ {0, 1}.
                    cxz_right_bits[x][hw] = carry as u8;
                }
                table.set_bytes(row_idx, cols::cxz_left(x, 0), &cxz_left_bytes[x]);
                table.set_bytes(row_idx, cols::cxz_right_bit(x, 0), &cxz_right_bits[x]);

                // Reconstruct: left[b] + (1 - b%2) * right[(b/2 + 3) mod 4]
                for b in 0..8 {
                    let right_contribution = match cols::cxz_right_bit_for_byte(b) {
                        Some(hw) => cxz_right_bits[x][hw],
                        None => 0,
                    };
                    rotated_c[x][b] = cxz_left_bytes[x][b].wrapping_add(right_contribution);
                }
            }

            // D[x] = C[(x-1)%5] XOR rotated_C[(x+1)%5]
            let mut d_bytes = [[0u8; 8]; 5];
            for x in 0..5 {
                for b in 0..8 {
                    let val = c_bytes[(x + 4) % 5][b] ^ rotated_c[(x + 1) % 5][b];
                    d_bytes[x][b] = val;
                }
                table.set_bytes(row_idx, cols::dxz(x, 0), &d_bytes[x]);
            }

            // theta[x][y] = start[x][y] XOR D[x]
            let mut theta_lanes = [0u64; 25];
            for x in 0..5 {
                for y in 0..5 {
                    let lane = state[x + 5 * y];
                    let mut d_lane = 0u64;
                    for b in 0..8 {
                        d_lane |= (d_bytes[x][b] as u64) << (b * 8);
                    }
                    theta_lanes[x + 5 * y] = lane ^ d_lane;
                    table.set_dword_bl(row_idx, cols::theta(x, y, 0), theta_lanes[x + 5 * y]);
                }
            }

            // === ρ (rho) ===
            // For each lane, rotate theta[x][y] by KECCAK_RHO[x][y] bits.
            // Decompose rotation as: rnc (nibble, 0..15) + 16*rbc[0] + 32*rbc[1].
            // rnc and rbc are inlined as compile-time constants per spec
            // [[variables.constant]]; only HWSL outputs are stored in the trace.
            for x in 0..5 {
                for y in 0..5 {
                    let rho_offset = KECCAK_RHO[x][y] as usize;
                    let rnc_val = (rho_offset % 16) as u8;
                    let theta_lane = theta_lanes[x + 5 * y];
                    let mut rot_left_bytes = [0u8; 8];
                    let mut rot_right_bytes = [0u8; 8];
                    for hw in 0..4 {
                        let halfword = ((theta_lane >> (hw * 16)) & 0xFFFF) as u16;
                        let (shifted, carry) = hwsl(halfword, rnc_val);
                        rot_left_bytes[hw * 2] = (shifted & 0xFF) as u8;
                        rot_left_bytes[hw * 2 + 1] = (shifted >> 8) as u8;
                        rot_right_bytes[hw * 2] = (carry & 0xFF) as u8;
                        rot_right_bytes[hw * 2 + 1] = (carry >> 8) as u8;
                    }
                    table.set_bytes(row_idx, cols::rot_left(x, y, 0), &rot_left_bytes);
                    table.set_bytes(row_idx, cols::rot_right(x, y, 0), &rot_right_bytes);
                }
            }

            // === π (pi) ===
            // pi[x][y] = rho[(x+3y)%5][x] where rho is the rotated theta.
            // pi is a spec [[variables.virtual]] — not stored as trace columns.
            // It's reconstructed inline in chi bus interactions as
            //   pi[x][y][z] = rot_left[sx,sy,l_byte] + rot_right[sx,sy,r_byte]
            // with (sx, sy) = ((x+3y)%5, x) and (l_byte, r_byte) resolved from
            // the compile-time rbc constant. pi_lanes is still computed here
            // for the chi step below.
            let mut pi_lanes = [0u64; 25];
            for x in 0..5 {
                for y in 0..5 {
                    let rotated = theta_lanes[x + 5 * y].rotate_left(KECCAK_RHO[x][y]);
                    let dst_x = y;
                    let dst_y = (2 * x + 3 * y) % 5;
                    pi_lanes[dst_x + 5 * dst_y] = rotated;
                }
            }

            // === χ (chi) ===
            let mut chi_lanes = [0u64; 25];
            for x in 0..5 {
                for y in 0..5 {
                    let not_next = !pi_lanes[(x + 1) % 5 + 5 * y];
                    let next2 = pi_lanes[(x + 2) % 5 + 5 * y];
                    let and_val = not_next & next2;
                    chi_lanes[x + 5 * y] = pi_lanes[x + 5 * y] ^ and_val;
                    table.set_dword_bl(row_idx, cols::chi_ands(x, y, 0), and_val);
                    table.set_dword_bl(row_idx, cols::chi(x, y, 0), chi_lanes[x + 5 * y]);
                }
            }

            // === ι (iota) ===
            let rc_val = KECCAK_RC[round];
            let iota_lane = chi_lanes[0] ^ rc_val;
            table.set_dword_bl(row_idx, cols::rc(0), rc_val);
            table.set_dword_bl(row_idx, cols::iota(0), iota_lane);

            // Update state for next round
            chi_lanes[0] = iota_lane;
            state = chi_lanes;

            // mu = 1 (real row)
            table.set_fe(row_idx, cols::MU, FE::one());
        }
    }

    // Padding rows have mu=0 and all zeros (default)
    trace
}

// =========================================================================
// Bus interactions (1,031 total)
// =========================================================================
//
// The θ/ρ halfword shifts no longer emit HWSL lookups (120 sends/row removed):
// they are enforced by the inline μ-gated linear identities in
// `KeccakRndConstraints`. The matching HWSL multiplicities are likewise dropped
// on the BITWISE side (`collect_bitwise_from_keccak`).

#[allow(clippy::needless_range_loop)]
pub fn bus_interactions() -> Vec<BusInteraction> {
    let mut interactions = Vec::with_capacity(1031);

    // --- IO group (3) ---

    // 1. KECCAK bus: receive (timestamp, round, start[200])
    // Per spec keccak_round.toml: input = ["timestamp", "round", "start"] where
    // start is [[[Byte, 8], 5], 5] — 200 Byte elements, each its own bus element.
    {
        let mut values = vec![
            BusValue::Packed {
                start_column: cols::TIMESTAMP_0,
                packing: Packing::Direct,
            },
            BusValue::Packed {
                start_column: cols::TIMESTAMP_1,
                packing: Packing::Direct,
            },
            BusValue::Packed {
                start_column: cols::ROUND,
                packing: Packing::Direct,
            },
        ];
        for x in 0..5 {
            for y in 0..5 {
                for b in 0..8 {
                    values.push(BusValue::Packed {
                        start_column: cols::start(x, y, b),
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

    // 2. KECCAK bus: send (timestamp, round+1, out[200])
    //    out[0][0] = iota, out[x][y] = chi for (x,y) != (0,0)
    {
        let mut values = vec![
            BusValue::Packed {
                start_column: cols::TIMESTAMP_0,
                packing: Packing::Direct,
            },
            BusValue::Packed {
                start_column: cols::TIMESTAMP_1,
                packing: Packing::Direct,
            },
            BusValue::linear(vec![
                LinearTerm::Column {
                    coefficient: 1,
                    column: cols::ROUND,
                },
                LinearTerm::Constant(1),
            ]),
        ];
        for x in 0..5 {
            for y in 0..5 {
                for b in 0..8 {
                    let col = if x == 0 && y == 0 {
                        cols::IOTA + b
                    } else {
                        cols::chi(x, y, b)
                    };
                    values.push(BusValue::Packed {
                        start_column: col,
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

    // 3. KECCAK_RC: lookup (round) → rc[8]
    {
        let mut values = vec![BusValue::Packed {
            start_column: cols::ROUND,
            packing: Packing::Direct,
        }];
        for b in 0..8 {
            values.push(BusValue::Packed {
                start_column: cols::rc(b),
                packing: Packing::Direct,
            });
        }
        interactions.push(BusInteraction::sender(
            BusId::KeccakRc,
            Multiplicity::Column(cols::MU),
            values,
        ));
    }

    // --- Theta: Cxz chain BYTE_ALU[XOR] (160) ---
    // Stage 0: XOR(start[x,0,z], start[x,1,z]) → Cxz[x,0,z]
    for x in 0..5 {
        for b in 0..8 {
            interactions.push(BusInteraction::sender(
                BusId::ByteAlu,
                Multiplicity::Column(cols::MU),
                vec![
                    BusValue::constant(alu_op::XOR as u64),
                    BusValue::Packed {
                        start_column: cols::start(x, 0, b),
                        packing: Packing::Direct,
                    },
                    BusValue::Packed {
                        start_column: cols::start(x, 1, b),
                        packing: Packing::Direct,
                    },
                    BusValue::Packed {
                        start_column: cols::cxz(x, 0, b),
                        packing: Packing::Direct,
                    },
                ],
            ));
        }
    }
    // Stages 1..3: XOR(Cxz[x,stage-1,z], start[x,stage+1,z]) → Cxz[x,stage,z]
    for x in 0..5 {
        for stage in 1..4usize {
            let y = stage + 1;
            for b in 0..8 {
                interactions.push(BusInteraction::sender(
                    BusId::ByteAlu,
                    Multiplicity::Column(cols::MU),
                    vec![
                        BusValue::constant(alu_op::XOR as u64),
                        BusValue::Packed {
                            start_column: cols::cxz(x, stage - 1, b),
                            packing: Packing::Direct,
                        },
                        BusValue::Packed {
                            start_column: cols::start(x, y, b),
                            packing: Packing::Direct,
                        },
                        BusValue::Packed {
                            start_column: cols::cxz(x, stage, b),
                            packing: Packing::Direct,
                        },
                    ],
                ));
            }
        }
    }

    // --- Theta: rotate-C-by-1 shift is enforced by an inline μ-gated linear
    //     identity (see `KeccakRndConstraints`), not an HWSL lookup. ---

    // --- Theta: ARE_BYTES range checks on Cxz_left (20 pairs) ---
    // Spec emits 40 `IS_BYTE<Cxz_left[x][z]>` templates; we merge adjacent
    // byte pairs (z=2i, z=2i+1) into ARE_BYTES interactions per the
    // implementation guidance in spec/is_byte.typ.
    // Cxz_right uses IS_BIT polynomial constraints (see `KeccakRndConstraints`).
    for x in 0..5 {
        for i in 0..4 {
            interactions.push(BusInteraction::sender(
                BusId::AreBytes,
                Multiplicity::Column(cols::MU),
                vec![
                    BusValue::Packed {
                        start_column: cols::cxz_left(x, 2 * i),
                        packing: Packing::Direct,
                    },
                    BusValue::Packed {
                        start_column: cols::cxz_left(x, 2 * i + 1),
                        packing: Packing::Direct,
                    },
                ],
            ));
        }
    }

    // --- Theta: Dxz BYTE_ALU[XOR] (40) ---
    // D[x][b] = C[(x-1)%5][b] XOR rotated_C[(x+1)%5][b]
    // rotated_C[x'][b] = Cxz_left[x'][b] + (1 - b%2) * Cxz_right[x'][(b/2 - 1)%4]
    // (spec d75944ee/9143370f). For odd b only Cxz_left contributes.
    for x in 0..5 {
        for b in 0..8 {
            let mut rotated_c_terms = vec![LinearTerm::Column {
                coefficient: 1,
                column: cols::cxz_left((x + 1) % 5, b),
            }];
            if let Some(hw) = cols::cxz_right_bit_for_byte(b) {
                rotated_c_terms.push(LinearTerm::Column {
                    coefficient: 1,
                    column: cols::cxz_right_bit((x + 1) % 5, hw),
                });
            }
            interactions.push(BusInteraction::sender(
                BusId::ByteAlu,
                Multiplicity::Column(cols::MU),
                vec![
                    BusValue::constant(alu_op::XOR as u64),
                    BusValue::Packed {
                        start_column: cols::cxz((x + 4) % 5, 3, b),
                        packing: Packing::Direct,
                    },
                    BusValue::linear(rotated_c_terms),
                    BusValue::Packed {
                        start_column: cols::dxz(x, b),
                        packing: Packing::Direct,
                    },
                ],
            ));
        }
    }

    // --- Theta final: BYTE_ALU[XOR] (200) ---
    // theta[x][y][b] = start[x][y][b] XOR D[x][b]
    for x in 0..5 {
        for y in 0..5 {
            for b in 0..8 {
                interactions.push(BusInteraction::sender(
                    BusId::ByteAlu,
                    Multiplicity::Column(cols::MU),
                    vec![
                        BusValue::constant(alu_op::XOR as u64),
                        BusValue::Packed {
                            start_column: cols::start(x, y, b),
                            packing: Packing::Direct,
                        },
                        BusValue::Packed {
                            start_column: cols::dxz(x, b),
                            packing: Packing::Direct,
                        },
                        BusValue::Packed {
                            start_column: cols::theta(x, y, b),
                            packing: Packing::Direct,
                        },
                    ],
                ));
            }
        }
    }

    // --- Rho: the per-lane shift is enforced by inline μ-gated linear
    //     identities (see `KeccakRndConstraints`), not HWSL lookups. ---

    // --- Rho: ARE_BYTES range checks on rot_left + rot_right (200 pairs) ---
    // Spec emits 400 IS_BYTE templates (200 per side); we merge each
    // (rot_left[x][y][b], rot_right[x][y][b]) into one ARE_BYTES interaction.
    for x in 0..5 {
        for y in 0..5 {
            for b in 0..8 {
                interactions.push(BusInteraction::sender(
                    BusId::AreBytes,
                    Multiplicity::Column(cols::MU),
                    vec![
                        BusValue::Packed {
                            start_column: cols::rot_left(x, y, b),
                            packing: Packing::Direct,
                        },
                        BusValue::Packed {
                            start_column: cols::rot_right(x, y, b),
                            packing: Packing::Direct,
                        },
                    ],
                ));
            }
        }
    }

    // --- Chi: BYTE_ALU[AND] (200) ---
    // chi_ands[x][y][b] = (255 - pi[(x+1)%5][y][b]) AND pi[(x+2)%5][y][b]
    // pi is virtual: pi[x][y][z] = rot_left[sx,sy,l_byte] + rot_right[sx,sy,r_byte]
    // with src lane (sx,sy) = ((x+3y)%5, x) and byte offsets from KECCAK_RHO.
    for x in 0..5 {
        for y in 0..5 {
            for b in 0..8 {
                let (p1_l, p1_r) = cols::pi_src_cols((x + 1) % 5, y, b);
                let (p2_l, p2_r) = cols::pi_src_cols((x + 2) % 5, y, b);
                interactions.push(BusInteraction::sender(
                    BusId::ByteAlu,
                    Multiplicity::Column(cols::MU),
                    vec![
                        BusValue::constant(alu_op::AND as u64),
                        BusValue::linear(vec![
                            LinearTerm::Constant(255),
                            LinearTerm::Column {
                                coefficient: -1,
                                column: p1_l,
                            },
                            LinearTerm::Column {
                                coefficient: -1,
                                column: p1_r,
                            },
                        ]),
                        BusValue::linear(vec![
                            LinearTerm::Column {
                                coefficient: 1,
                                column: p2_l,
                            },
                            LinearTerm::Column {
                                coefficient: 1,
                                column: p2_r,
                            },
                        ]),
                        BusValue::Packed {
                            start_column: cols::chi_ands(x, y, b),
                            packing: Packing::Direct,
                        },
                    ],
                ));
            }
        }
    }

    // --- Chi: BYTE_ALU[XOR] (200) ---
    // chi[x][y][b] = pi[x][y][b] XOR chi_ands[x][y][b] (pi virtual).
    for x in 0..5 {
        for y in 0..5 {
            for b in 0..8 {
                let (p_l, p_r) = cols::pi_src_cols(x, y, b);
                interactions.push(BusInteraction::sender(
                    BusId::ByteAlu,
                    Multiplicity::Column(cols::MU),
                    vec![
                        BusValue::constant(alu_op::XOR as u64),
                        BusValue::linear(vec![
                            LinearTerm::Column {
                                coefficient: 1,
                                column: p_l,
                            },
                            LinearTerm::Column {
                                coefficient: 1,
                                column: p_r,
                            },
                        ]),
                        BusValue::Packed {
                            start_column: cols::chi_ands(x, y, b),
                            packing: Packing::Direct,
                        },
                        BusValue::Packed {
                            start_column: cols::chi(x, y, b),
                            packing: Packing::Direct,
                        },
                    ],
                ));
            }
        }
    }

    // --- Iota: BYTE_ALU[XOR] (8) ---
    // iota[b] = chi[0][0][b] XOR rc[b]
    for b in 0..8 {
        interactions.push(BusInteraction::sender(
            BusId::ByteAlu,
            Multiplicity::Column(cols::MU),
            vec![
                BusValue::constant(alu_op::XOR as u64),
                BusValue::Packed {
                    start_column: cols::chi(0, 0, b),
                    packing: Packing::Direct,
                },
                BusValue::Packed {
                    start_column: cols::rc(b),
                    packing: Packing::Direct,
                },
                BusValue::Packed {
                    start_column: cols::iota(b),
                    packing: Packing::Direct,
                },
            ],
        ));
    }

    interactions
}

// =========================================================================
// Single-source constraint set (ConstraintBuilder front-end)
// =========================================================================

/// The 16-bit value `main[lo_col] + 256·main[hi_col]` (byte pair → halfword).
#[inline]
fn halfword<B: ConstraintBuilder<GoldilocksField, GoldilocksExtension>>(
    b: &B,
    lo_col: usize,
    hi_col: usize,
) -> B::Expr {
    b.main(0, lo_col) + b.main(0, hi_col) * b.const_base(256)
}

/// The KECCAK round table's 140 transition constraints as a single
/// [`ConstraintSet`]:
///
/// * **20 IS_BIT** on the θ carry bits: for `x ∈ 0..5`, `hw ∈ 0..4`, the μ-gated
///   `μ · Cxz_right·(1 − Cxz_right)` (degree 3). Load-bearing: it pins the θ
///   carry to a single bit so the θ shift identity below is unique.
/// * **20 θ shift identities** (rnc = 1): for `x ∈ 0..5`, `hw ∈ 0..4`,
///   `μ · (in·2 − right·2¹⁶ − left)` where `in` is the `Cxz[x][3]` halfword,
///   `left` the `Cxz_left` byte pair and `right` the single `Cxz_right` carry
///   bit (degree 2).
/// * **100 ρ shift identities**: for `x,y ∈ 0..5`, `hw ∈ 0..4` with
///   `rnc = KECCAK_RHO[x][y] % 16`, `μ · (in·2^rnc − right·2¹⁶ − left)` where
///   `in` is the `theta[x][y]` halfword, `left`/`right` the `rot_left`/
///   `rot_right` byte pairs (degree 2; the general form covers rnc = 0, which
///   pins right = 0, left = in).
///
/// These identities replace the former θ/ρ HWSL bus lookups. Uniqueness of the
/// (left, right) decomposition rests on the ARE_BYTES range checks bounding both
/// halves to `[0, 2¹⁶)` and on `2¹⁶` being invertible mod the Goldilocks prime
/// (z3-verified equivalent to the HWSL contract).
#[derive(Clone, Copy)]
pub struct KeccakRndConstraints;

impl ConstraintSet<GoldilocksField, GoldilocksExtension> for KeccakRndConstraints {
    // The IS_BIT constraints are gated by μ (cond·x·(1−x)), so degree 3; the
    // shift identities are μ × linear, degree 2.
    fn max_degree(&self) -> usize {
        3
    }

    #[allow(clippy::needless_range_loop)]
    fn eval<B: ConstraintBuilder<GoldilocksField, GoldilocksExtension>>(&self, b: &mut B) {
        use crate::constraints::templates::emit_is_bit;

        let two_16 = 1u64 << 16;
        let mut idx = 0;

        // (1) IS_BIT on the θ carry bits (Cxz_right).
        for x in 0..5 {
            for hw in 0..4 {
                emit_is_bit(b, idx, cols::cxz_right_bit(x, hw), Some(cols::MU));
                idx += 1;
            }
        }

        // (2) θ rotate-C-by-1 shift identity (rnc = 1):
        //     μ · (in·2 − right·2¹⁶ − left) = 0.
        for x in 0..5 {
            for hw in 0..4 {
                let inp = halfword(b, cols::cxz(x, 3, hw * 2), cols::cxz(x, 3, hw * 2 + 1));
                let left = halfword(b, cols::cxz_left(x, hw * 2), cols::cxz_left(x, hw * 2 + 1));
                let right = b.main(0, cols::cxz_right_bit(x, hw));
                let identity = inp * b.const_base(2) - right * b.const_base(two_16) - left;
                let mu = b.main(0, cols::MU);
                b.emit_base(idx, mu * identity);
                idx += 1;
            }
        }

        // (3) ρ shift identity (rnc = KECCAK_RHO[x][y] % 16):
        //     μ · (in·2^rnc − right·2¹⁶ − left) = 0.
        for x in 0..5 {
            for y in 0..5 {
                let rnc = KECCAK_RHO[x][y] % 16;
                let pow = 1u64 << rnc;
                for hw in 0..4 {
                    let inp = halfword(b, cols::theta(x, y, hw * 2), cols::theta(x, y, hw * 2 + 1));
                    let left = halfword(
                        b,
                        cols::rot_left(x, y, hw * 2),
                        cols::rot_left(x, y, hw * 2 + 1),
                    );
                    let right = halfword(
                        b,
                        cols::rot_right(x, y, hw * 2),
                        cols::rot_right(x, y, hw * 2 + 1),
                    );
                    let identity = inp * b.const_base(pow) - right * b.const_base(two_16) - left;
                    let mu = b.main(0, cols::MU);
                    b.emit_base(idx, mu * identity);
                    idx += 1;
                }
            }
        }
    }
}
