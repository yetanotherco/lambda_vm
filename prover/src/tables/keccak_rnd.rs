//! KECCAK_RND: Round chip for Keccak-f[1600] permutation.
//!
//! One row per round (24 rows per keccak call). All bitwise operations are
//! delegated to BITWISE lookup tables (XOR_BYTE, AND_BYTE, HWSL, IS_BYTE).
//!
//! ## Column layout (1,456 columns)
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
//! | rot_left       |  192 | Left half of ρ rotation, 24 non-(0,0) lanes [8]   |
//! | rot_right      |  192 | Right half of ρ rotation, 24 non-(0,0) lanes [8]  |
//! | chi_ands       |  200 | AND results for χ [5][5][8]                       |
//! | chi            |  200 | State after χ [5][5][8]                           |
//! | rc             |    4 | Non-zero round constant bytes (positions 0,1,3,7) |
//! | iota           |    4 | χ[0][0][b] ⊕ rc[b] for b ∈ {0,1,3,7}               |
//! | mu             |    1 | Multiplicity (1 for real, 0 for padding)          |
//!
//! Note: spec [[variables.constant]] `rnc` and `rbc` are inlined as compile-time
//! constants derived from `KECCAK_RHO[x][y]`, not materialized as columns.
//! `Cxz_right` is typed `[Bit, 4]` per spec d75944ee — HWSL with shift=1
//! produces a single-bit carry, range-checked via IS_BIT polynomial constraints.
//! `rc[2,4,5,6]` are constant zero across all 24 round constants (LFSR-derived,
//! bits only at positions {0,1,3,7,15,31,63}); not stored, and ι aliases
//! chi[0][0][b] for those bytes (spec keccak.typ optimization note).
//! ρ is the identity on lane (0,0) (`KECCAK_RHO[0][0] = 0`), so `rot_left[0][0]`
//! and `rot_right[0][0]` are not stored; π for that lane references
//! `theta[0][0]` directly (spec keccak.typ optimization note).

use executor::vm::instruction::execution::{KECCAK_RC, KECCAK_RHO};
use stark::constraints::transition::{TransitionConstraint, TransitionConstraintEvaluator};
use stark::lookup::{BusInteraction, BusValue, LinearTerm, Multiplicity, Packing};
use stark::trace::TraceTable;

use super::types::{BusId, FE, GoldilocksExtension, GoldilocksField};

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

    // rot_left[24][8] = 192 bytes (lane (0,0) skipped; ρ is identity there).
    pub const ROT_LEFT: usize = THETA + 200; // 663

    // rot_right[24][8] = 192 bytes
    pub const ROT_RIGHT: usize = ROT_LEFT + 192; // 855

    // chi_ands[5][5][8] = 200 bytes
    // (pi is a spec [[variables.virtual]] — inlined as rot_left + rot_right at
    // compile-resolved offsets, not materialized as columns.)
    pub const CHI_ANDS: usize = ROT_RIGHT + 192; // 1047

    // chi[5][5][8] = 200 bytes — state after χ
    pub const CHI: usize = CHI_ANDS + 200; // 1247

    // rc[4] — non-zero round constant bytes (positions 0, 1, 3, 7).
    // Index `i ∈ 0..4` corresponds to byte position `RC_NONZERO_BYTES[i]`.
    pub const RC: usize = CHI + 200; // 1447

    // iota[4] — χ[0][0][b] ⊕ rc[b] for b ∈ {0, 1, 3, 7}.
    // For b ∈ {2, 4, 5, 6}, ι output equals χ[0][0][b] (rc[b] = 0).
    pub const IOTA: usize = RC + 4; // 1451

    // mu — multiplicity flag.
    // rnc and rbc (spec [[variables.constant]]) are inlined as compile-time
    // constants from KECCAK_RHO, not allocated as columns.
    pub const MU: usize = IOTA + 4; // 1455

    pub const NUM_COLUMNS: usize = MU + 1; // 1456

    /// Byte positions of the round constant that are non-zero across all 24
    /// rounds (the others are constant zero and not stored).
    pub const RC_NONZERO_BYTES: [usize; 4] = [0, 1, 3, 7];

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

    /// Lane offset for `rot_left`/`rot_right`. Lane (0,0) is skipped because
    /// `KECCAK_RHO[0][0] = 0` (ρ is the identity there), so we avoid storing
    /// 16 always-redundant bytes (8 in rot_left + 8 in rot_right). Only valid
    /// for lanes other than (0,0).
    #[inline]
    const fn rot_lane_offset(x: usize, y: usize) -> usize {
        let lane = x + 5 * y;
        debug_assert!(lane != 0, "rot_left/rot_right not stored for lane (0,0)");
        (lane - 1) * 8
    }

    /// Index into `rot_left[x][y][byte]`. Panics for lane (0,0).
    #[inline]
    pub const fn rot_left(x: usize, y: usize, byte: usize) -> usize {
        ROT_LEFT + rot_lane_offset(x, y) + byte
    }

    /// Index into `rot_right[x][y][byte]`. Panics for lane (0,0).
    #[inline]
    pub const fn rot_right(x: usize, y: usize, byte: usize) -> usize {
        ROT_RIGHT + rot_lane_offset(x, y) + byte
    }

    /// Source-lane info for resolving `pi[x][y][z]` (spec virtual).
    ///
    /// - For source lane `(0,0)` returns `PiSource::Theta(theta_col)`: ρ is the
    ///   identity there (`KECCAK_RHO[0][0] = 0`) so π takes θ directly.
    /// - For all other source lanes returns
    ///   `PiSource::RotPair { left_col, right_col }`: the usual HWSL split.
    #[inline]
    pub fn pi_src(x: usize, y: usize, z: usize) -> PiSource {
        use executor::vm::instruction::execution::KECCAK_RHO;
        let sx = (x + 3 * y) % 5;
        let sy = x;
        if sx == 0 && sy == 0 {
            return PiSource::Theta(theta(0, 0, z));
        }
        let rho_offset = KECCAK_RHO[sx][sy] as usize;
        let rbc_val = rho_offset / 16;
        let (l_byte, r_byte) = match rbc_val {
            0 => (z, (z + 6) % 8),
            1 => ((z + 6) % 8, (z + 4) % 8),
            2 => ((z + 4) % 8, (z + 2) % 8),
            3 => ((z + 2) % 8, z),
            _ => unreachable!(),
        };
        PiSource::RotPair {
            left_col: rot_left(sx, sy, l_byte),
            right_col: rot_right(sx, sy, r_byte),
        }
    }

    /// Linear-term resolution of `pi[x][y][z]` (spec virtual variable).
    ///
    /// For lane (0,0) it's a single `theta` column; for all other lanes it's
    /// the sum of one `rot_left` and one `rot_right` column.
    #[derive(Clone, Copy, Debug)]
    pub enum PiSource {
        /// Single column: `theta(0, 0, z)` (ρ is identity for lane (0,0)).
        Theta(usize),
        /// Two columns whose sum equals `pi[x][y][z]`.
        RotPair { left_col: usize, right_col: usize },
    }

    /// Index into chi_ands[x][y][byte]
    #[inline]
    pub const fn chi_ands(x: usize, y: usize, byte: usize) -> usize {
        CHI_ANDS + (x + 5 * y) * 8 + byte
    }

    /// Index into chi[x][y][byte]
    #[inline]
    pub const fn chi(x: usize, y: usize, byte: usize) -> usize {
        CHI + (x * 5 + y) * 8 + byte
    }

    /// Index into rc[i] for `i ∈ 0..4`. `i` maps to actual byte position
    /// `RC_NONZERO_BYTES[i]` ∈ {0, 1, 3, 7}.
    #[inline]
    pub const fn rc(i: usize) -> usize {
        RC + i
    }

    /// Index into iota[i] for `i ∈ 0..4`. `i` maps to actual byte position
    /// `RC_NONZERO_BYTES[i]` ∈ {0, 1, 3, 7}.
    #[inline]
    pub const fn iota(i: usize) -> usize {
        IOTA + i
    }

    /// For byte `b ∈ 0..8`, return `Some(i)` if `b` is a stored non-zero rc
    /// byte (so `rc(i)` / `iota(i)` are valid); else `None` (rc[b] is the
    /// constant zero and `iota[0,0,b] == chi[0,0,b]`).
    #[inline]
    pub const fn rc_index_for_byte(b: usize) -> Option<usize> {
        match b {
            0 => Some(0),
            1 => Some(1),
            3 => Some(2),
            7 => Some(3),
            _ => None,
        }
    }

    /// Column representing the post-ι output byte at lane `(x, y)`, byte `b`.
    /// For lane (0,0) and `b ∈ {0,1,3,7}` this is `iota(rc_index_for_byte(b))`;
    /// for lane (0,0) and `b ∈ {2,4,5,6}` it aliases `chi(0, 0, b)` (since
    /// `iota[0,0,b] = chi[0,0,b] XOR 0 = chi[0,0,b]`); for other lanes it is
    /// `chi(x, y, b)`.
    #[inline]
    pub const fn output_byte_col(x: usize, y: usize, b: usize) -> usize {
        if x == 0 && y == 0 {
            match rc_index_for_byte(b) {
                Some(i) => iota(i),
                None => chi(0, 0, b),
            }
        } else {
            chi(x, y, b)
        }
    }
}

// =========================================================================
// pi (spec virtual) helpers
// =========================================================================

/// Linear-term decomposition of the spec virtual variable `pi[x][y][z]`,
/// each term with coefficient 1. Resolves to a single `theta(0,0,z)` column
/// for source lane (0,0) (ρ is identity), or to a `(rot_left, rot_right)`
/// pair otherwise.
fn pi_terms(x: usize, y: usize, z: usize) -> Vec<LinearTerm> {
    match cols::pi_src(x, y, z) {
        cols::PiSource::Theta(c) => vec![LinearTerm::Column {
            coefficient: 1,
            column: c,
        }],
        cols::PiSource::RotPair {
            left_col,
            right_col,
        } => vec![
            LinearTerm::Column {
                coefficient: 1,
                column: left_col,
            },
            LinearTerm::Column {
                coefficient: 1,
                column: right_col,
            },
        ],
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
            data[base + cols::TIMESTAMP_0] = FE::from(op.timestamp & 0xFFFF_FFFF);
            data[base + cols::TIMESTAMP_1] = FE::from(op.timestamp >> 32);
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
            // Lane (0,0) has rnc=0 (ρ is identity); rot_left/rot_right are
            // not stored and π references theta[0][0] directly.
            for x in 0..5 {
                for y in 0..5 {
                    if x == 0 && y == 0 {
                        continue;
                    }
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
            // Only the 4 non-zero rc bytes (positions 0, 1, 3, 7) are stored.
            // For the zero bytes {2, 4, 5, 6}, ι is the identity on chi[0][0].
            let rc_val = KECCAK_RC[round];
            for (i, &b) in cols::RC_NONZERO_BYTES.iter().enumerate() {
                data[base + cols::rc(i)] = FE::from(byte_of(rc_val, b) as u64);
                let iota_byte = byte_of(chi_lanes[0], b) ^ byte_of(rc_val, b);
                data[base + cols::iota(i)] = FE::from(iota_byte as u64);
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
// Bus interactions (1,411 total)
// =========================================================================

#[allow(clippy::needless_range_loop)]
pub fn bus_interactions() -> Vec<BusInteraction> {
    let mut interactions = Vec::with_capacity(1380);

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
                    values.push(BusValue::Packed {
                        start_column: cols::output_byte_col(x, y, b),
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

    // 3. KECCAK_RC: lookup (round) → rc[0], rc[1], rc[3], rc[7]
    //    (the other 4 bytes are constant zero across all rounds).
    {
        let mut values = vec![BusValue::Packed {
            start_column: cols::ROUND,
            packing: Packing::Direct,
        }];
        for i in 0..4 {
            values.push(BusValue::Packed {
                start_column: cols::rc(i),
                packing: Packing::Direct,
            });
        }
        interactions.push(BusInteraction::sender(
            BusId::KeccakRc,
            Multiplicity::Column(cols::MU),
            values,
        ));
    }

    // --- Theta: Cxz chain XOR_BYTE (160) ---
    // Stage 0: XOR(start[x,0,z], start[x,1,z]) → Cxz[x,0,z]
    for x in 0..5 {
        for b in 0..8 {
            interactions.push(BusInteraction::sender(
                BusId::XorByte,
                Multiplicity::Column(cols::MU),
                vec![
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
                    BusId::XorByte,
                    Multiplicity::Column(cols::MU),
                    vec![
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

    // --- Theta: IS_BYTE range checks on Cxz_left (40) ---
    // Cxz_right uses IS_BIT polynomial constraints (see create_constraints).
    for x in 0..5 {
        for b in 0..8 {
            interactions.push(BusInteraction::sender(
                BusId::IsByte,
                Multiplicity::Column(cols::MU),
                vec![BusValue::Packed {
                    start_column: cols::cxz_left(x, b),
                    packing: Packing::Direct,
                }],
            ));
        }
    }

    // --- Theta: Dxz XOR_BYTE (40) ---
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
                BusId::XorByte,
                Multiplicity::Column(cols::MU),
                vec![
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

    // --- Theta final: XOR_BYTE (200) ---
    // theta[x][y][b] = start[x][y][b] XOR D[x][b]
    for x in 0..5 {
        for y in 0..5 {
            for b in 0..8 {
                interactions.push(BusInteraction::sender(
                    BusId::XorByte,
                    Multiplicity::Column(cols::MU),
                    vec![
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

    // --- Rho: HWSL (96) ---
    // HWSL(theta[x][y] halfword[hw], rnc[x][y]) → (rot_left, rot_right).
    // Lane (0,0) is skipped: KECCAK_RHO[0][0] = 0 makes ρ the identity.
    for x in 0..5 {
        for y in 0..5 {
            if x == 0 && y == 0 {
                continue;
            }
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

    // --- Rho: IS_BYTE range checks on rot_left + rot_right (384) ---
    // Lane (0,0) skipped (no rot_left/rot_right stored).
    for x in 0..5 {
        for y in 0..5 {
            if x == 0 && y == 0 {
                continue;
            }
            for b in 0..8 {
                interactions.push(BusInteraction::sender(
                    BusId::IsByte,
                    Multiplicity::Column(cols::MU),
                    vec![BusValue::Packed {
                        start_column: cols::rot_left(x, y, b),
                        packing: Packing::Direct,
                    }],
                ));
                interactions.push(BusInteraction::sender(
                    BusId::IsByte,
                    Multiplicity::Column(cols::MU),
                    vec![BusValue::Packed {
                        start_column: cols::rot_right(x, y, b),
                        packing: Packing::Direct,
                    }],
                ));
            }
        }
    }

    // --- Chi: AND_BYTE (200) ---
    // chi_ands[x][y][b] = (255 - pi[(x+1)%5][y][b]) AND pi[(x+2)%5][y][b]
    // pi is virtual: usually rot_left[sx,sy,l_byte] + rot_right[sx,sy,r_byte],
    // but for source lane (0,0) it is theta[0][0][b] directly (ρ is identity).
    for x in 0..5 {
        for y in 0..5 {
            for b in 0..8 {
                let p1_terms = pi_terms((x + 1) % 5, y, b);
                let p2_terms = pi_terms((x + 2) % 5, y, b);
                let mut not_p1 = vec![LinearTerm::Constant(255)];
                for term in &p1_terms {
                    if let LinearTerm::Column {
                        coefficient,
                        column,
                    } = *term
                    {
                        not_p1.push(LinearTerm::Column {
                            coefficient: -coefficient,
                            column,
                        });
                    }
                }
                interactions.push(BusInteraction::sender(
                    BusId::AndByte,
                    Multiplicity::Column(cols::MU),
                    vec![
                        BusValue::linear(not_p1),
                        BusValue::linear(p2_terms),
                        BusValue::Packed {
                            start_column: cols::chi_ands(x, y, b),
                            packing: Packing::Direct,
                        },
                    ],
                ));
            }
        }
    }

    // --- Chi: XOR_BYTE (200) ---
    // chi[x][y][b] = pi[x][y][b] XOR chi_ands[x][y][b] (pi virtual).
    for x in 0..5 {
        for y in 0..5 {
            for b in 0..8 {
                interactions.push(BusInteraction::sender(
                    BusId::XorByte,
                    Multiplicity::Column(cols::MU),
                    vec![
                        BusValue::linear(pi_terms(x, y, b)),
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

    // --- Iota: XOR_BYTE (4) ---
    // Only the 4 non-zero rc bytes need an XOR; for b ∈ {2, 4, 5, 6}, rc = 0
    // so iota[0,0,b] = chi[0,0,b] (handled directly in the output sender via
    // `output_byte_col`).
    for (i, &b) in cols::RC_NONZERO_BYTES.iter().enumerate() {
        interactions.push(BusInteraction::sender(
            BusId::XorByte,
            Multiplicity::Column(cols::MU),
            vec![
                BusValue::Packed {
                    start_column: cols::chi(0, 0, b),
                    packing: Packing::Direct,
                },
                BusValue::Packed {
                    start_column: cols::rc(i),
                    packing: Packing::Direct,
                },
                BusValue::Packed {
                    start_column: cols::iota(i),
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
/// All other checks (XOR, AND, HWSL, IS_BYTE, IS_HALF, KECCAK, KECCAK_RC) are
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

#[cfg(test)]
mod tests {
    use super::*;
    use executor::vm::instruction::execution::keccak_f1600;

    /// pi is a spec virtual variable. Verify the inlined expression
    /// (rot_left[sx,sy,l_byte] + rot_right[sx,sy,r_byte]) matches the byte of
    /// rho(theta) for a non-trivial state. Uses mu=0 padding rows as a trivial
    /// sanity check (all zeros), then a non-zero-input round as the real test.
    #[test]
    fn test_pi_virtual_matches_rotate() {
        // Use a non-zero input so theta_lanes are non-trivial.
        let input = [0x0102030405060708u64; 25];
        let mut output = input;
        keccak_f1600(&mut output);
        let op = KeccakRoundOperation {
            timestamp: 42,
            input,
            output,
        };
        let trace = generate_keccak_rnd_trace(&[op]);
        let base = 0;

        // Recompute theta for round 0 in u64 to compare against virtual pi.
        let mut c = [0u64; 5];
        for x in 0..5 {
            c[x] = input[x] ^ input[x + 5] ^ input[x + 10] ^ input[x + 15] ^ input[x + 20];
        }
        let mut d = [0u64; 5];
        for x in 0..5 {
            d[x] = c[(x + 4) % 5] ^ c[(x + 1) % 5].rotate_left(1);
        }
        let mut theta_lanes = [0u64; 25];
        for x in 0..5 {
            for y in 0..5 {
                theta_lanes[x + 5 * y] = input[x + 5 * y] ^ d[x];
            }
        }

        for x in 0..5 {
            for y in 0..5 {
                let sx = (x + 3 * y) % 5;
                let sy = x;
                let rotated = theta_lanes[sx + 5 * sy].rotate_left(KECCAK_RHO[sx][sy]);
                for z in 0..8 {
                    let virtual_pi = match cols::pi_src(x, y, z) {
                        cols::PiSource::Theta(c) => trace.main_table.data[base + c],
                        cols::PiSource::RotPair {
                            left_col,
                            right_col,
                        } => {
                            &trace.main_table.data[base + left_col]
                                + &trace.main_table.data[base + right_col]
                        }
                    };
                    let expected = FE::from((rotated >> (z * 8)) & 0xFF);
                    assert_eq!(
                        virtual_pi, expected,
                        "virtual pi mismatch at ({x},{y},{z}): sx={sx}, sy={sy}"
                    );
                }
            }
        }
    }
}
