//! KECCAK_RND: Round chip for Keccak-f[1600] permutation.
//!
//! One row per round (24 rows per keccak call). All bitwise operations are
//! delegated to BITWISE lookup tables (BYTE_ALU, HWSL, ARE_BYTES).
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
//! `Cxz_right` is typed `[Bit, 4]` per spec d75944ee — HWSL with shift=1
//! produces a single-bit carry, range-checked via IS_BIT polynomial constraints.

use executor::vm::instruction::execution::{KECCAK_RC, KECCAK_RHO};
use stark::constraints::transition::{TransitionConstraint, TransitionConstraintEvaluator};
use stark::lookup::{BusInteraction, BusValue, LinearTerm, Multiplicity, Packing};
use stark::trace::TraceTable;

use super::limbs::set_limbs_32;
use super::types::{BusId, FE, GoldilocksExtension, GoldilocksField, alu_op};

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
    let n_rows = (ops.len() * 24).next_power_of_two().max(4);
    let mut data = vec![FE::zero(); n_rows * cols::NUM_COLUMNS];

    for (op_idx, op) in ops.iter().enumerate() {
        // Execute round-by-round, tracking the state
        let mut state = op.input;

        for round in 0..24 {
            let row_idx = op_idx * 24 + round;
            let base = row_idx * cols::NUM_COLUMNS;

            // Timestamp & round
            set_limbs_32(&mut data, base + cols::TIMESTAMP_0, op.timestamp);
            data[base + cols::ROUND] = FE::from(round as u64);

            // start = current state as bytes
            for x in 0..5 {
                for y in 0..5 {
                    let lane = state[x + 5 * y];
                    for b in 0..8 {
                        data[base + cols::start(x, y, b)] = FE::from(byte_of(lane, b) as u64);
                    }
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
                    data[base + cols::cxz(x, 0, b)] = FE::from(cxz[x][0][b] as u64);
                }
                // Stages 1..3: XOR(Cxz[x][k-1], start[x, k+1])
                for stage in 1..4 {
                    let y = stage + 1;
                    for b in 0..8 {
                        let prev = cxz[x][stage - 1][b];
                        let sv = byte_of(state[x + 5 * y], b);
                        cxz[x][stage][b] = prev ^ sv;
                        data[base + cols::cxz(x, stage, b)] = FE::from(cxz[x][stage][b] as u64);
                    }
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
                    data[base + cols::cxz_left(x, hw * 2)] =
                        FE::from(cxz_left_bytes[x][hw * 2] as u64);
                    data[base + cols::cxz_left(x, hw * 2 + 1)] =
                        FE::from(cxz_left_bytes[x][hw * 2 + 1] as u64);
                    data[base + cols::cxz_right_bit(x, hw)] =
                        FE::from(cxz_right_bits[x][hw] as u64);
                }
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
                    data[base + cols::dxz(x, b)] = FE::from(val as u64);
                }
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
                    for b in 0..8 {
                        data[base + cols::theta(x, y, b)] =
                            FE::from(byte_of(theta_lanes[x + 5 * y], b) as u64);
                    }
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
                    for hw in 0..4 {
                        let halfword = ((theta_lane >> (hw * 16)) & 0xFFFF) as u16;
                        let (shifted, carry) = hwsl(halfword, rnc_val);
                        data[base + cols::rot_left(x, y, hw * 2)] =
                            FE::from((shifted & 0xFF) as u64);
                        data[base + cols::rot_left(x, y, hw * 2 + 1)] =
                            FE::from((shifted >> 8) as u64);
                        data[base + cols::rot_right(x, y, hw * 2)] =
                            FE::from((carry & 0xFF) as u64);
                        data[base + cols::rot_right(x, y, hw * 2 + 1)] =
                            FE::from((carry >> 8) as u64);
                    }
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
                    for b in 0..8 {
                        data[base + cols::chi_ands(x, y, b)] = FE::from(byte_of(and_val, b) as u64);
                        data[base + cols::chi(x, y, b)] =
                            FE::from(byte_of(chi_lanes[x + 5 * y], b) as u64);
                    }
                }
            }

            // === ι (iota) ===
            let rc_val = KECCAK_RC[round];
            for b in 0..8 {
                data[base + cols::rc(b)] = FE::from(byte_of(rc_val, b) as u64);
                let iota_byte = byte_of(chi_lanes[0], b) ^ byte_of(rc_val, b);
                data[base + cols::iota(b)] = FE::from(iota_byte as u64);
            }

            // Update state for next round
            chi_lanes[0] ^= rc_val;
            state = chi_lanes;

            // mu = 1 (real row)
            data[base + cols::MU] = FE::one();
        }
    }

    // Padding rows have mu=0 and all zeros (default)
    TraceTable::new_main(data, cols::NUM_COLUMNS, 1)
}

// =========================================================================
// Bus interactions (1,371 total)
// =========================================================================

#[allow(clippy::needless_range_loop)]
pub fn bus_interactions() -> Vec<BusInteraction> {
    let mut interactions = Vec::with_capacity(1371);

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

    // --- Theta: HWSL for rotated C (20) ---
    // HWSL(C[x] halfword[hw], 1) → (Cxz_left, Cxz_right)
    // Cxz_right is a single carry bit zero-extended to a halfword (spec d75944ee).
    for x in 0..5 {
        for hw in 0..4 {
            interactions.push(BusInteraction::sender(
                BusId::Hwsl,
                Multiplicity::Column(cols::MU),
                vec![
                    // Input halfword: Cxz[x][3][hw*2] + 256 * Cxz[x][3][hw*2+1]
                    BusValue::linear(vec![
                        LinearTerm::Column {
                            coefficient: 1,
                            column: cols::cxz(x, 3, hw * 2),
                        },
                        LinearTerm::Column {
                            coefficient: 256,
                            column: cols::cxz(x, 3, hw * 2 + 1),
                        },
                    ]),
                    // Shift amount = 1
                    BusValue::constant(1),
                    // Output: shifted
                    BusValue::linear(vec![
                        LinearTerm::Column {
                            coefficient: 1,
                            column: cols::cxz_left(x, hw * 2),
                        },
                        LinearTerm::Column {
                            coefficient: 256,
                            column: cols::cxz_left(x, hw * 2 + 1),
                        },
                    ]),
                    // Output: carry (single bit cast to Half — high byte = 0).
                    BusValue::Packed {
                        start_column: cols::cxz_right_bit(x, hw),
                        packing: Packing::Direct,
                    },
                ],
            ));
        }
    }

    // --- Theta: ARE_BYTES range checks on Cxz_left (20 pairs) ---
    // Spec emits 40 `IS_BYTE<Cxz_left[x][z]>` templates; we merge adjacent
    // byte pairs (z=2i, z=2i+1) into ARE_BYTES interactions per the
    // implementation guidance in spec/is_byte.typ.
    // Cxz_right uses IS_BIT polynomial constraints (see create_constraints).
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

    // --- Rho: HWSL (100) ---
    // HWSL(theta[x][y] halfword[hw], rnc[x][y]) → (rot_left, rot_right)
    // rnc is inlined as a constant: KECCAK_RHO[x][y] % 16.
    for x in 0..5 {
        for y in 0..5 {
            let rnc_val = (KECCAK_RHO[x][y] % 16) as u64;
            for hw in 0..4 {
                interactions.push(BusInteraction::sender(
                    BusId::Hwsl,
                    Multiplicity::Column(cols::MU),
                    vec![
                        BusValue::linear(vec![
                            LinearTerm::Column {
                                coefficient: 1,
                                column: cols::theta(x, y, hw * 2),
                            },
                            LinearTerm::Column {
                                coefficient: 256,
                                column: cols::theta(x, y, hw * 2 + 1),
                            },
                        ]),
                        BusValue::constant(rnc_val),
                        BusValue::linear(vec![
                            LinearTerm::Column {
                                coefficient: 1,
                                column: cols::rot_left(x, y, hw * 2),
                            },
                            LinearTerm::Column {
                                coefficient: 256,
                                column: cols::rot_left(x, y, hw * 2 + 1),
                            },
                        ]),
                        BusValue::linear(vec![
                            LinearTerm::Column {
                                coefficient: 1,
                                column: cols::rot_right(x, y, hw * 2),
                            },
                            LinearTerm::Column {
                                coefficient: 256,
                                column: cols::rot_right(x, y, hw * 2 + 1),
                            },
                        ]),
                    ],
                ));
            }
        }
    }

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
// Constraints
// =========================================================================

/// KECCAK_RND polynomial constraints: 20 IS_BIT(μ; Cxz_right) constraints.
///
/// Per spec d75944ee, `Cxz_right` is typed `[Bit, 4], 5` and range-checked via
/// IS_BIT polynomial constraints (kind="template", cond="μ"), not lookups:
///   μ * Cxz_right[x][hw] * (1 - Cxz_right[x][hw]) = 0
///
/// - pi is a spec [[variables.virtual]] inlined in chi bus interactions.
/// - rnc/rbc are spec [[variables.constant]] inlined as compile-time constants.
///
/// All other checks (XOR, AND, HWSL, ARE_BYTES, IS_HALF, KECCAK, KECCAK_RC) are
/// enforced via bus interactions against the BITWISE/KECCAK_RC chips.
pub fn create_constraints(
    constraint_idx_start: usize,
) -> (
    Vec<Box<dyn TransitionConstraintEvaluator<GoldilocksField, GoldilocksExtension>>>,
    usize,
) {
    use crate::constraints::templates::IsBitConstraint;

    let mut constraints: Vec<
        Box<dyn TransitionConstraintEvaluator<GoldilocksField, GoldilocksExtension>>,
    > = Vec::with_capacity(20);
    let mut idx = constraint_idx_start;
    for x in 0..5 {
        for hw in 0..4 {
            constraints
                .push(IsBitConstraint::new(cols::MU, cols::cxz_right_bit(x, hw), idx).boxed());
            idx += 1;
        }
    }
    (constraints, idx)
}
