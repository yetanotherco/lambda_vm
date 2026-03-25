//! Keccak-f[1600] permutation chip.
//!
//! A single wide table (2,638 columns) that proves Keccak-f[1600] permutations.
//! Each permutation uses 24 rows (one per round). All intermediate values are
//! stored as columns and verified via purely polynomial constraints — no lookups.
//!
//! ## Design
//!
//! Ported from Plonky3's keccak-air approach: bit-level decomposition of the
//! Keccak state allows expressing theta, rho, pi, chi, and iota as degree-1
//! to degree-3 polynomial constraints.
//!
//! ## Column Layout (3,139 columns)
//!
//! - `step_flags[24]`: One-hot round selector
//! - `first`: First-round real-row flag (= step_flags[0] * mu)
//! - `export`: Final round flag (= step_flags[23] * mu)
//! - `preimage[5][5][4]`: Input state in 16-bit limbs (100 cols)
//! - `preimage_bytes[25][8]`: Byte view of the input state (200 cols)
//! - `a[5][5][4]`: State after theta input (100 cols)
//! - `c[5][64]`: Column parities (320 cols)
//! - `c_prime[5][64]`: Rotated column parities (320 cols)
//! - `a_prime[5][5][64]`: Theta output in bits (1,600 cols)
//! - `a_prime_prime[5][5][4]`: After chi in 16-bit limbs (100 cols)
//! - `a_prime_prime_0_0_bits[64]`: Lane [0,0] bit decomposition (64 cols)
//! - `a_prime_prime_prime_0_0_limbs[4]`: Lane [0,0] after iota (4 cols)
//! - `lane_addr[25]`: DWordHL address for each 8-byte lane (100 cols)
//! - `output_bytes[25][8]`: Byte view of the current round output (200 cols)
//! - `mu`: Multiplicity flag (1 col)
//!
//! ## Bus Interactions
//!
//! The table binds one input snapshot and one output snapshot per permutation:
//! - `EcallKeccak` receive on `first`
//! - 25 `Memw` reads on `first`
//! - 25 `Memw` writes on `export`
//!
//! ## Constraints
//!
//! Constraints follow the Plonky3 keccak-air structure: boolean/range checks,
//! theta/chi/iota relations, and row-to-row consistency across rounds.

use math::field::element::FieldElement;
use math::field::traits::{IsField, IsSubFieldOf};
use stark::constraints::transition::TransitionConstraint;
use stark::lookup::{BusInteraction, BusValue, LinearTerm, Multiplicity, Packing};
use stark::table::TableView;
use stark::trace::TraceTable;
use stark::traits::TransitionEvaluationContext;

use super::types::{BusId, FE, GoldilocksExtension, GoldilocksField};
use crate::constraints::templates::{AddConstraint, AddOperand};

use executor::vm::instruction::execution::KECCAK_SYSCALL_NUMBER;

// =============================================================================
// Keccak constants
// =============================================================================

/// Round constants for Keccak-f[1600] (24 rounds).
pub const KECCAK_RC: [u64; 24] = [
    0x0000000000000001,
    0x0000000000008082,
    0x800000000000808A,
    0x8000000080008000,
    0x000000000000808B,
    0x0000000080000001,
    0x8000000080008081,
    0x8000000000008009,
    0x000000000000008A,
    0x0000000000000088,
    0x0000000080008009,
    0x000000008000000A,
    0x000000008000808B,
    0x800000000000008B,
    0x8000000000008089,
    0x8000000000008003,
    0x8000000000008002,
    0x8000000000000080,
    0x000000000000800A,
    0x800000008000000A,
    0x8000000080008081,
    0x8000000000008080,
    0x0000000080000001,
    0x8000000080008008,
];

/// Rotation offsets R[x][y] for the rho step of Keccak-f[1600].
pub const RHO: [[u32; 5]; 5] = [
    [0, 36, 3, 41, 18],
    [1, 44, 10, 45, 2],
    [62, 6, 43, 15, 61],
    [28, 55, 25, 21, 56],
    [27, 20, 39, 8, 14],
];

// =============================================================================
// Column indices
// =============================================================================

pub mod cols {
    /// Round one-hot flags: step_flags[i] = 1 iff this is round i.
    pub const STEP_FLAGS: usize = 0;
    pub const STEP_FLAGS_END: usize = STEP_FLAGS + 24;

    /// First-round real-row flag: 1 only on round 0 of real permutations.
    pub const FIRST: usize = STEP_FLAGS_END;

    /// Export flag: 1 on the final round (round 23) of a real permutation.
    pub const EXPORT: usize = FIRST + 1;

    /// Preimage: 25 lanes × 4 limbs (16-bit each) = 100 columns.
    /// Layout: preimage[y][x][limb] at offset PREIMAGE + (y*5 + x)*4 + limb.
    pub const PREIMAGE: usize = EXPORT + 1;
    pub const PREIMAGE_END: usize = PREIMAGE + 100;

    /// Preimage bytes: 25 lanes × 8 bytes = 200 columns.
    pub const PREIMAGE_BYTES: usize = PREIMAGE_END;
    pub const PREIMAGE_BYTES_END: usize = PREIMAGE_BYTES + 200;

    /// State A (theta input): same layout as preimage.
    pub const A: usize = PREIMAGE_BYTES_END;
    pub const A_END: usize = A + 100;

    /// Column parities C[x][z]: 5 columns × 64 bits = 320 columns.
    pub const C: usize = A_END;
    pub const C_END: usize = C + 320;

    /// Rotated column parities C'[x][z]: 5 columns × 64 bits = 320 columns.
    pub const C_PRIME: usize = C_END;
    pub const C_PRIME_END: usize = C_PRIME + 320;

    /// Theta output A'[x][y][z] in bits: 5×5×64 = 1600 columns.
    pub const A_PRIME: usize = C_PRIME_END;
    pub const A_PRIME_END: usize = A_PRIME + 1600;

    /// Chi output A''[x][y] in 16-bit limbs: 5×5×4 = 100 columns.
    pub const A_PRIME_PRIME: usize = A_PRIME_END;
    pub const A_PRIME_PRIME_END: usize = A_PRIME_PRIME + 100;

    /// Bit decomposition of A''[0][0]: 64 bits.
    pub const A_PRIME_PRIME_0_0_BITS: usize = A_PRIME_PRIME_END;
    pub const A_PRIME_PRIME_0_0_BITS_END: usize = A_PRIME_PRIME_0_0_BITS + 64;

    /// A'''[0][0] after iota: 4 limbs (16-bit each).
    pub const A_PRIME_PRIME_PRIME_0_0_LIMBS: usize = A_PRIME_PRIME_0_0_BITS_END;
    pub const A_PRIME_PRIME_PRIME_0_0_LIMBS_END: usize = A_PRIME_PRIME_PRIME_0_0_LIMBS + 4;

    /// Multiplicity: 1 for real rows, 0 for padding.
    pub const MU: usize = A_PRIME_PRIME_PRIME_0_0_LIMBS_END;

    /// Timestamp (DWordWL: 2 columns, lo32 and hi32).
    /// Same timestamp for all 24 rows of a permutation (from the CPU ECALL).
    pub const TIMESTAMP_0: usize = MU + 1;
    pub const TIMESTAMP_1: usize = TIMESTAMP_0 + 1;

    /// State address (DWordWL: 2 columns, lo32 and hi32).
    /// Base address of the 200-byte state in memory.
    pub const STATE_ADDR_0: usize = TIMESTAMP_1 + 1;
    pub const STATE_ADDR_1: usize = STATE_ADDR_0 + 1;

    /// Per-lane addresses as DWordHL (4 halfwords each): state_addr + 8 * lane_idx.
    pub const LANE_ADDR_START: usize = STATE_ADDR_1 + 1;

    /// Output bytes for the current round output: 25 lanes × 8 bytes = 200 columns.
    pub const OUTPUT_BYTES: usize = LANE_ADDR_START + 100;
    pub const OUTPUT_BYTES_END: usize = OUTPUT_BYTES + 200;

    /// Total number of columns.
    pub const NUM_COLUMNS: usize = OUTPUT_BYTES_END;

    /// Per-lane address columns as DWordHL (4 halfwords).
    pub fn lane_addr(lane_idx: usize) -> [usize; 4] {
        let base = LANE_ADDR_START + lane_idx * 4;
        [base, base + 1, base + 2, base + 3]
    }
}

// Verify column count at compile time
const _: () = assert!(cols::NUM_COLUMNS == 3139);

// =============================================================================
// Column index helpers
// =============================================================================

/// Index into preimage/a/a_prime_prime arrays: [y][x][limb]
#[inline]
const fn lane_limb_idx(x: usize, y: usize, limb: usize) -> usize {
    (y * 5 + x) * 4 + limb
}

/// Index into byte arrays: [y][x][byte]
#[inline]
const fn lane_byte_idx(x: usize, y: usize, byte: usize) -> usize {
    (y * 5 + x) * 8 + byte
}

/// Index into c/c_prime arrays: [x][z]
#[inline]
const fn parity_idx(x: usize, z: usize) -> usize {
    x * 64 + z
}

/// Index into a_prime array: [y][x][z]
#[inline]
const fn a_prime_idx(x: usize, y: usize, z: usize) -> usize {
    (y * 5 + x) * 64 + z
}

// =============================================================================
// Keccak operation
// =============================================================================

/// A single Keccak-f[1600] permutation to be proven.
#[derive(Debug, Clone)]
pub struct KeccakOperation {
    /// Timestamp from the CPU ECALL.
    pub timestamp: u64,
    /// Address of the 200-byte state in memory.
    pub state_addr: u64,
    /// Input state (25 × u64, little-endian).
    pub input: [u64; 25],
    /// Output state after keccak-f[1600].
    pub output: [u64; 25],
}

// =============================================================================
// Trace generation
// =============================================================================

/// Extract 16-bit limb `limb_idx` from a u64 lane.
#[inline]
fn limb16(lane: u64, limb_idx: usize) -> u64 {
    (lane >> (limb_idx * 16)) & 0xFFFF
}

/// Extract bit `z` from a u64 lane.
#[inline]
fn bit(lane: u64, z: usize) -> u64 {
    (lane >> z) & 1
}

/// Generate the Keccak trace table from a list of permutation operations.
///
/// Each operation produces 24 rows (one per Keccak round).
/// Padding rows have mu=0 and follow valid dummy zero-input rounds so all
/// transition constraints remain satisfied.
fn fill_round(
    data: &mut [FE],
    row_idx: usize,
    round: usize,
    preimage: &[u64; 25],
    a: &[u64; 25],
    timestamp: u64,
    state_addr: u64,
    is_real: bool,
) -> [u64; 25] {
    let base = row_idx * cols::NUM_COLUMNS;

    // Step flags (one-hot)
    data[base + cols::STEP_FLAGS + round] = FE::one();

    // First row only for real permutations.
    if is_real && round == 0 {
        data[base + cols::FIRST] = FE::one();
    }

    // Export only on the final round of a real permutation.
    if is_real && round == 23 {
        data[base + cols::EXPORT] = FE::one();
    }

    // Mu = 1 for real rows, 0 for padding.
    if is_real {
        data[base + cols::MU] = FE::one();
    }

    // Timestamp and state address are attached to every row in the permutation.
    data[base + cols::TIMESTAMP_0] = FE::from(timestamp & 0xFFFF_FFFF);
    data[base + cols::TIMESTAMP_1] = FE::from(timestamp >> 32);
    data[base + cols::STATE_ADDR_0] = FE::from(state_addr & 0xFFFF_FFFF);
    data[base + cols::STATE_ADDR_1] = FE::from(state_addr >> 32);

    // Preimage is constant across a permutation. A is the current round input.
    for y in 0..5 {
        for x in 0..5 {
            let preimage_lane = preimage[x + 5 * y];
            let a_lane = a[x + 5 * y];
            for limb in 0..4 {
                data[base + cols::PREIMAGE + lane_limb_idx(x, y, limb)] =
                    FE::from(limb16(preimage_lane, limb));
                data[base + cols::A + lane_limb_idx(x, y, limb)] = FE::from(limb16(a_lane, limb));
            }
            for byte in 0..8 {
                data[base + cols::PREIMAGE_BYTES + lane_byte_idx(x, y, byte)] =
                    FE::from((preimage_lane >> (byte * 8)) & 0xFF);
            }
        }
    }

    for lane_idx in 0..25 {
        let addr = state_addr.wrapping_add(lane_idx as u64 * 8);
        let addr_cols = cols::lane_addr(lane_idx);
        data[base + addr_cols[0]] = FE::from(addr & 0xFFFF);
        data[base + addr_cols[1]] = FE::from((addr >> 16) & 0xFFFF);
        data[base + addr_cols[2]] = FE::from((addr >> 32) & 0xFFFF);
        data[base + addr_cols[3]] = FE::from((addr >> 48) & 0xFFFF);
    }

    // === THETA ===
    // C[x][z] = XOR over y of A[x][y][z]
    let mut c = [[0u64; 64]; 5];
    for x in 0..5 {
        for z in 0..64 {
            let mut parity = 0u64;
            for y in 0..5 {
                parity ^= bit(a[x + 5 * y], z);
            }
            c[x][z] = parity;
            data[base + cols::C + parity_idx(x, z)] = FE::from(parity);
        }
    }

    // Plonky3 convention:
    // C'[x][z] = C[x][z] XOR C[(x-1) mod 5][z] XOR C[(x+1) mod 5][(z-1) mod 64]
    let mut c_prime = [[0u64; 64]; 5];
    for x in 0..5 {
        for z in 0..64 {
            let val = c[x][z] ^ c[(x + 4) % 5][z] ^ c[(x + 1) % 5][(z + 63) % 64];
            c_prime[x][z] = val;
            data[base + cols::C_PRIME + parity_idx(x, z)] = FE::from(val);
        }
    }

    // A'[x][y][z] = A[x][y][z] XOR C[x][z] XOR C'[x][z]
    let mut a_prime = [[[0u64; 64]; 5]; 5];
    for y in 0..5 {
        for x in 0..5 {
            for z in 0..64 {
                let val = bit(a[x + 5 * y], z) ^ c[x][z] ^ c_prime[x][z];
                a_prime[x][y][z] = val;
                data[base + cols::A_PRIME + a_prime_idx(x, y, z)] = FE::from(val);
            }
        }
    }

    // === RHO + PI ===
    // B[y][(2x+3y) mod 5][z] = A'[x][y][(z - R[x][y]) mod 64]
    let mut b = [[[0u64; 64]; 5]; 5];
    for x in 0..5 {
        for y in 0..5 {
            let dst_x = y;
            let dst_y = (2 * x + 3 * y) % 5;
            let rot = RHO[x][y] as usize;
            for z in 0..64 {
                b[dst_x][dst_y][z] = a_prime[x][y][(z + 64 - rot) % 64];
            }
        }
    }

    // === CHI ===
    // A''[x][y][z] = B[x][y][z] XOR ((NOT B[(x+1)%5][y][z]) AND B[(x+2)%5][y][z])
    let mut a_pp_lanes = [0u64; 25];
    for y in 0..5 {
        for x in 0..5 {
            let mut lane = 0u64;
            #[allow(clippy::needless_range_loop)]
            for z in 0..64 {
                let chi_bit = b[x][y][z] ^ ((1 - b[(x + 1) % 5][y][z]) & b[(x + 2) % 5][y][z]);
                lane |= chi_bit << z;
            }
            a_pp_lanes[x + 5 * y] = lane;
            for limb in 0..4 {
                data[base + cols::A_PRIME_PRIME + lane_limb_idx(x, y, limb)] =
                    FE::from(limb16(lane, limb));
            }
        }
    }

    // Bit decomposition of A''[0][0]
    let a_pp_0_0 = a_pp_lanes[0];
    for z in 0..64 {
        data[base + cols::A_PRIME_PRIME_0_0_BITS + z] = FE::from(bit(a_pp_0_0, z));
    }

    // === IOTA ===
    // A'''[0][0] = A''[0][0] XOR RC[round]
    let a_ppp_0_0 = a_pp_0_0 ^ KECCAK_RC[round];
    for limb in 0..4 {
        data[base + cols::A_PRIME_PRIME_PRIME_0_0_LIMBS + limb] = FE::from(limb16(a_ppp_0_0, limb));
    }

    // Compute output state for this round
    let mut output = a_pp_lanes;
    output[0] = a_ppp_0_0;
    for y in 0..5 {
        for x in 0..5 {
            let lane = output[x + 5 * y];
            for byte in 0..8 {
                data[base + cols::OUTPUT_BYTES + lane_byte_idx(x, y, byte)] =
                    FE::from((lane >> (byte * 8)) & 0xFF);
            }
        }
    }
    output
}

pub fn generate_keccak_trace(
    ops: &[KeccakOperation],
) -> TraceTable<GoldilocksField, GoldilocksExtension> {
    let n_real_rows = ops.len() * 24;
    let num_rows = n_real_rows.next_power_of_two().max(4);
    let mut data = vec![FE::zero(); num_rows * cols::NUM_COLUMNS];

    for (perm_idx, op) in ops.iter().enumerate() {
        let mut state = op.input;
        let preimage = op.input;

        for round in 0..24 {
            let row_idx = perm_idx * 24 + round;
            state = fill_round(
                &mut data,
                row_idx,
                round,
                &preimage,
                &state,
                op.timestamp,
                op.state_addr,
                true,
            );
        }
    }

    // Fill padding rows with valid zero-input dummy rounds.
    let mut row_idx = n_real_rows;
    while row_idx < num_rows {
        let preimage = [0u64; 25];
        let mut state = preimage;
        let rounds = (num_rows - row_idx).min(24);

        for round in 0..rounds {
            state = fill_round(&mut data, row_idx, round, &preimage, &state, 0, 0, false);
            row_idx += 1;
        }
    }

    TraceTable::new_main(data, cols::NUM_COLUMNS, 1)
}

// =============================================================================
// Constraints
// =============================================================================

/// Keccak constraint kinds.
#[derive(Debug, Clone, Copy)]
enum KeccakConstraintKind {
    /// sum(step_flags) = 1
    StepFlagsSumOne,
    /// first = step_flags[0] * mu
    FirstMatchesStep0Mu,
    /// step_flags[i] = next.step_flags[(i+1) mod 24]
    StepFlagRotation(usize),
    /// step_flags[0] * (preimage[x][y][limb] - a[x][y][limb]) = 0
    FirstStepPreimageEqualsA(usize),
    /// (1 - step_flags[23]) * (preimage - next.preimage) = 0
    PreimageConsistency(usize),
    /// (1 - step_flags[23]) * (col - next.col) = 0
    TransitionConsistency(usize),
    /// first * (preimage_limb - byte0 - 256*byte1) = 0
    InputByteLimb(usize),
    /// export * (output_limb - byte0 - 256*byte1) = 0
    OutputByteLimb { x: usize, y: usize, limb: usize },
    /// export = step_flags[23] * mu
    ExportMatchesFinalStep,
    /// c'[x][z] = c[x][z] XOR c[(x-1)%5][z] XOR c[(x+1)%5][(z-1)%64]
    ThetaCPrime { x: usize, z: usize },
    /// sum_y a'[x][y][z] - c'[x][z] ∈ {0, 2, 4}
    ThetaParity { x: usize, z: usize },
    /// a''[x][y][limb] = sum of chi bits * 2^k (degree 3)
    ChiLimb { x: usize, y: usize, limb: usize },
    /// a''[0][0][limb] = sum of a_pp_0_0_bits * 2^k
    APP00LimbFromBits(usize),
    /// a'''[0][0][limb] = XOR(a''[0][0], RC[round]) selected by step_flags
    IotaLimb(usize),
    /// a[x][y][limb] = sum of (a'[x][y][z] XOR c[x][z] XOR c'[x][z]) * 2^k
    /// This links A (limbs) to A' (bits) via theta.
    ALimbFromAPrime { x: usize, y: usize, limb: usize },
    /// (1 - step_flags[23]) * (output_limb - next.a_limb) = 0
    NextAFromOutput { x: usize, y: usize, limb: usize },
}

struct KeccakConstraint {
    kind: KeccakConstraintKind,
    constraint_idx: usize,
}

impl KeccakConstraint {
    #[inline]
    fn xor2<F>(a: &FieldElement<F>, b: &FieldElement<F>) -> FieldElement<F>
    where
        F: IsField,
    {
        let two = FieldElement::<F>::from(2u64);
        a.clone() + b.clone() - two * a.clone() * b.clone()
    }

    #[inline]
    fn xor3<F>(a: &FieldElement<F>, b: &FieldElement<F>, c: &FieldElement<F>) -> FieldElement<F>
    where
        F: IsField,
    {
        let ab = Self::xor2(a, b);
        Self::xor2(&ab, c)
    }

    #[allow(clippy::assign_op_pattern)]
    fn compute<F, E>(
        &self,
        local: &TableView<F, E>,
        next: Option<&TableView<F, E>>,
    ) -> FieldElement<F>
    where
        F: IsSubFieldOf<E>,
        E: IsField,
    {
        let get =
            |col: usize| -> FieldElement<F> { local.get_main_evaluation_element(0, col).clone() };
        let get_next = |col: usize| -> FieldElement<F> {
            next.expect("missing next row")
                .get_main_evaluation_element(0, col)
                .clone()
        };
        let one = FieldElement::<F>::one();

        match self.kind {
            KeccakConstraintKind::StepFlagsSumOne => {
                let mut sum = FieldElement::<F>::zero();
                for i in 0..24 {
                    sum = sum + get(cols::STEP_FLAGS + i);
                }
                sum - one
            }
            KeccakConstraintKind::FirstMatchesStep0Mu => {
                get(cols::FIRST) - get(cols::STEP_FLAGS) * get(cols::MU)
            }
            KeccakConstraintKind::StepFlagRotation(i) => {
                get(cols::STEP_FLAGS + i) - get_next(cols::STEP_FLAGS + ((i + 1) % 24))
            }
            KeccakConstraintKind::FirstStepPreimageEqualsA(flat_idx) => {
                get(cols::STEP_FLAGS) * (get(cols::PREIMAGE + flat_idx) - get(cols::A + flat_idx))
            }
            KeccakConstraintKind::PreimageConsistency(flat_idx) => {
                let not_final = one.clone() - get(cols::STEP_FLAGS + 23);
                not_final * (get(cols::PREIMAGE + flat_idx) - get_next(cols::PREIMAGE + flat_idx))
            }
            KeccakConstraintKind::TransitionConsistency(col) => {
                let not_final = one.clone() - get(cols::STEP_FLAGS + 23);
                not_final * (get(col) - get_next(col))
            }
            KeccakConstraintKind::InputByteLimb(flat_idx) => {
                let lane_idx = flat_idx / 4;
                let limb = flat_idx % 4;
                let x = lane_idx % 5;
                let y = lane_idx / 5;
                let byte0 = get(cols::PREIMAGE_BYTES + lane_byte_idx(x, y, limb * 2));
                let byte1 = get(cols::PREIMAGE_BYTES + lane_byte_idx(x, y, limb * 2 + 1));
                let limb_val = get(cols::PREIMAGE + flat_idx);
                get(cols::FIRST) * (limb_val - byte0 - FieldElement::<F>::from(256u64) * byte1)
            }
            KeccakConstraintKind::OutputByteLimb { x, y, limb } => {
                let byte0 = get(cols::OUTPUT_BYTES + lane_byte_idx(x, y, limb * 2));
                let byte1 = get(cols::OUTPUT_BYTES + lane_byte_idx(x, y, limb * 2 + 1));
                let output_limb = if x == 0 && y == 0 {
                    get(cols::A_PRIME_PRIME_PRIME_0_0_LIMBS + limb)
                } else {
                    get(cols::A_PRIME_PRIME + lane_limb_idx(x, y, limb))
                };
                get(cols::EXPORT) * (output_limb - byte0 - FieldElement::<F>::from(256u64) * byte1)
            }
            KeccakConstraintKind::ExportMatchesFinalStep => {
                get(cols::EXPORT) - get(cols::STEP_FLAGS + 23) * get(cols::MU)
            }
            KeccakConstraintKind::ThetaCPrime { x, z } => {
                // c'[x][z] = c[x][z] XOR c[(x-1)%5][z] XOR c[(x+1)%5][(z-1)%64]
                let c_self = get(cols::C + parity_idx(x, z));
                let c_left = get(cols::C + parity_idx((x + 4) % 5, z));
                let c_right = get(cols::C + parity_idx((x + 1) % 5, (z + 63) % 64));
                let cp = get(cols::C_PRIME + parity_idx(x, z));
                cp - Self::xor3(&c_self, &c_left, &c_right)
            }
            KeccakConstraintKind::ThetaParity { x, z } => {
                let mut sum = FieldElement::<F>::zero();
                for y in 0..5 {
                    sum = sum + get(cols::A_PRIME + a_prime_idx(x, y, z));
                }
                let diff = sum - get(cols::C_PRIME + parity_idx(x, z));
                diff.clone()
                    * (diff.clone() - FieldElement::<F>::from(2u64))
                    * (diff - FieldElement::<F>::from(4u64))
            }
            KeccakConstraintKind::ChiLimb { x, y, limb } => {
                // Compute pi^{-1}: b[bx][by] comes from a'[sx][sy] with rotation
                // sx = (3*by + bx) % 5, sy = bx, rot = RHO[sx][sy]
                let src_x = |bx: usize, by: usize| (3 * by + bx) % 5;
                let src_y = |bx: usize, _by: usize| bx;

                let b_bit = |bx: usize, by: usize, z: usize| -> FieldElement<F> {
                    let sx = src_x(bx, by);
                    let sy = src_y(bx, by);
                    let r = RHO[sx][sy] as usize;
                    get(cols::A_PRIME + a_prime_idx(sx, sy, (z + 64 - r) % 64))
                };

                let mut limb_val = FieldElement::<F>::zero();
                for k in 0..16 {
                    let z = limb * 16 + k;
                    let bxy = b_bit(x, y, z);
                    let bx1y = b_bit((x + 1) % 5, y, z);
                    let bx2y = b_bit((x + 2) % 5, y, z);

                    // chi: bxy XOR ((!bx1y) AND bx2y) = bxy + (1-bx1y)*bx2y - 2*bxy*(1-bx1y)*bx2y
                    let not_b1_and_b2 = (one.clone() - bx1y) * bx2y.clone();
                    let chi_bit = Self::xor2(&bxy, &not_b1_and_b2);

                    let coeff = FieldElement::<F>::from(1u64 << k);
                    limb_val = limb_val + chi_bit * coeff;
                }

                get(cols::A_PRIME_PRIME + lane_limb_idx(x, y, limb)) - limb_val
            }
            KeccakConstraintKind::APP00LimbFromBits(limb) => {
                let mut sum = FieldElement::<F>::zero();
                for k in 0..16 {
                    let bit_val = get(cols::A_PRIME_PRIME_0_0_BITS + limb * 16 + k);
                    let coeff = FieldElement::<F>::from(1u64 << k);
                    sum = sum + bit_val * coeff;
                }
                get(cols::A_PRIME_PRIME + lane_limb_idx(0, 0, limb)) - sum
            }
            KeccakConstraintKind::IotaLimb(limb) => {
                // a'''[0][0][limb] = sum over rounds: step_flags[r] * XOR_limb(a_pp_0_0, RC[r])
                let mut iota_limb = FieldElement::<F>::zero();
                for (round, &rc) in KECCAK_RC.iter().enumerate() {
                    let flag = get(cols::STEP_FLAGS + round);
                    let mut round_limb = FieldElement::<F>::zero();
                    for k in 0..16 {
                        let z = limb * 16 + k;
                        let a_bit = get(cols::A_PRIME_PRIME_0_0_BITS + z);
                        let rc_bit_val = (rc >> z) & 1;
                        let rc_bit = FieldElement::<F>::from(rc_bit_val);
                        let xor_bit = Self::xor2(&a_bit, &rc_bit);
                        let coeff = FieldElement::<F>::from(1u64 << k);
                        round_limb = round_limb + xor_bit * coeff;
                    }
                    iota_limb = iota_limb + flag * round_limb;
                }
                get(cols::A_PRIME_PRIME_PRIME_0_0_LIMBS + limb) - iota_limb
            }
            KeccakConstraintKind::ALimbFromAPrime { x, y, limb } => {
                // a[x][y][limb] = sum_{k=0}^{15} (a'[x][y][z] XOR c[x][z] XOR c'[x][z]) * 2^k
                let mut sum = FieldElement::<F>::zero();
                for k in 0..16 {
                    let z = limb * 16 + k;
                    let ap = get(cols::A_PRIME + a_prime_idx(x, y, z));
                    let c = get(cols::C + parity_idx(x, z));
                    let cp = get(cols::C_PRIME + parity_idx(x, z));
                    let a_bit = Self::xor3(&ap, &c, &cp);
                    let coeff = FieldElement::<F>::from(1u64 << k);
                    sum = sum + a_bit * coeff;
                }
                get(cols::A + lane_limb_idx(x, y, limb)) - sum
            }
            KeccakConstraintKind::NextAFromOutput { x, y, limb } => {
                let not_final = one - get(cols::STEP_FLAGS + 23);
                let output = if x == 0 && y == 0 {
                    get(cols::A_PRIME_PRIME_PRIME_0_0_LIMBS + limb)
                } else {
                    get(cols::A_PRIME_PRIME + lane_limb_idx(x, y, limb))
                };
                let next_a = get_next(cols::A + lane_limb_idx(x, y, limb));
                not_final * (output - next_a)
            }
        }
    }
}

impl TransitionConstraint<GoldilocksField, GoldilocksExtension> for KeccakConstraint {
    fn degree(&self) -> usize {
        match self.kind {
            KeccakConstraintKind::StepFlagsSumOne => 1,
            KeccakConstraintKind::FirstMatchesStep0Mu => 2,
            KeccakConstraintKind::StepFlagRotation(_) => 1,
            KeccakConstraintKind::FirstStepPreimageEqualsA(_) => 2,
            KeccakConstraintKind::PreimageConsistency(_) => 2,
            KeccakConstraintKind::TransitionConsistency(_) => 2,
            KeccakConstraintKind::InputByteLimb(_) => 2,
            KeccakConstraintKind::OutputByteLimb { .. } => 2,
            KeccakConstraintKind::ExportMatchesFinalStep => 2,
            KeccakConstraintKind::ThetaCPrime { .. } => 3,
            KeccakConstraintKind::ThetaParity { .. } => 3,
            KeccakConstraintKind::ChiLimb { .. } => 3,
            KeccakConstraintKind::APP00LimbFromBits(_) => 1,
            KeccakConstraintKind::IotaLimb(_) => 2,
            KeccakConstraintKind::ALimbFromAPrime { .. } => 3,
            KeccakConstraintKind::NextAFromOutput { .. } => 2,
        }
    }

    fn constraint_idx(&self) -> usize {
        self.constraint_idx
    }

    fn end_exemptions(&self) -> usize {
        match self.kind {
            KeccakConstraintKind::StepFlagRotation(_)
            | KeccakConstraintKind::PreimageConsistency(_)
            | KeccakConstraintKind::TransitionConsistency(_)
            | KeccakConstraintKind::NextAFromOutput { .. } => 1,
            _ => 0,
        }
    }

    fn evaluate(
        &self,
        evaluation_context: &TransitionEvaluationContext<GoldilocksField, GoldilocksExtension>,
        transition_evaluations: &mut [FieldElement<GoldilocksExtension>],
    ) {
        match evaluation_context {
            TransitionEvaluationContext::Prover {
                frame,
                periodic_values: _,
                rap_challenges: _,
                ..
            } => {
                let next = if self.end_exemptions() > 0 {
                    Some(frame.get_evaluation_step(1))
                } else {
                    None
                };
                let val = self.compute(frame.get_evaluation_step(0), next);
                transition_evaluations[self.constraint_idx] = val.to_extension();
            }
            TransitionEvaluationContext::Verifier {
                frame,
                periodic_values: _,
                rap_challenges: _,
                ..
            } => {
                let next = if self.end_exemptions() > 0 {
                    Some(frame.get_evaluation_step(1))
                } else {
                    None
                };
                let val = self.compute(frame.get_evaluation_step(0), next);
                transition_evaluations[self.constraint_idx] = val;
            }
        }
    }
}

/// Create all Keccak constraints. Returns constraints and the next available index.
///
/// Constraint groups:
/// - IS_BIT: step_flags(24), first(1), export(1), mu(1), c(320), c'(320), a'(1600), a_pp_0_0_bits(64) = 2,331
/// - Step flags sum to one: 1
/// - first matches step_flags[0] * mu: 1
/// - Step flag rotation: 24 (transition)
/// - First-step preimage = A: 100
/// - Preimage consistency: 100 (transition)
/// - Timestamp/state-address consistency: 4 (transition)
/// - Input byte ↔ limb consistency: 100
/// - Output byte ↔ limb consistency: 100
/// - Export matches final-step real row: 1
/// - Theta C' (xor3): 320
/// - Theta parity link: 320
/// - A limb from A' (theta link): 100
/// - Chi limb: 100
/// - A''[0][0] limb from bits: 4
/// - Iota limb: 4
/// - Next-round A linkage: 100 (transition)
/// - lane_addr = state_addr + 8*lane_idx: 50
///
/// Total: 3,660
pub fn create_constraints(
    constraint_idx_start: usize,
) -> (
    Vec<Box<dyn TransitionConstraint<GoldilocksField, GoldilocksExtension>>>,
    usize,
) {
    let mut constraints: Vec<Box<dyn TransitionConstraint<GoldilocksField, GoldilocksExtension>>> =
        Vec::new();
    let mut idx = constraint_idx_start;

    // --- IS_BIT constraints (via template) ---

    // Collect all boolean column indices
    let mut bit_cols: Vec<usize> = Vec::with_capacity(2331);

    // step_flags[0..24]
    for i in 0..24 {
        bit_cols.push(cols::STEP_FLAGS + i);
    }
    // first
    bit_cols.push(cols::FIRST);
    // export
    bit_cols.push(cols::EXPORT);
    // mu
    bit_cols.push(cols::MU);
    // c[5][64] = 320
    for i in 0..320 {
        bit_cols.push(cols::C + i);
    }
    // c'[5][64] = 320
    for i in 0..320 {
        bit_cols.push(cols::C_PRIME + i);
    }
    // a'[5][5][64] = 1600
    for i in 0..1600 {
        bit_cols.push(cols::A_PRIME + i);
    }
    // a_pp_0_0_bits[64]
    for i in 0..64 {
        bit_cols.push(cols::A_PRIME_PRIME_0_0_BITS + i);
    }

    let (is_bit_constraints, next) =
        crate::constraints::templates::new_is_bit_constraints(&bit_cols, idx);
    for c in is_bit_constraints {
        constraints.push(Box::new(c));
    }
    idx = next;

    macro_rules! add {
        ($kind:expr) => {{
            constraints.push(Box::new(KeccakConstraint {
                kind: $kind,
                constraint_idx: idx,
            }));
            idx += 1;
        }};
    }

    // --- Step flags sum to one (1 constraint) ---
    add!(KeccakConstraintKind::StepFlagsSumOne);

    // --- first = step_flags[0] * mu (1 constraint) ---
    add!(KeccakConstraintKind::FirstMatchesStep0Mu);

    // --- Step flag rotation (24 constraints) ---
    for i in 0..24 {
        add!(KeccakConstraintKind::StepFlagRotation(i));
    }

    // --- First-step preimage = A (100 constraints) ---
    for flat_idx in 0..100 {
        add!(KeccakConstraintKind::FirstStepPreimageEqualsA(flat_idx));
    }

    // --- Preimage consistency (100 constraints) ---
    for flat_idx in 0..100 {
        add!(KeccakConstraintKind::PreimageConsistency(flat_idx));
    }

    // --- Timestamp and state address consistency (4 constraints) ---
    for col in [
        cols::TIMESTAMP_0,
        cols::TIMESTAMP_1,
        cols::STATE_ADDR_0,
        cols::STATE_ADDR_1,
    ] {
        add!(KeccakConstraintKind::TransitionConsistency(col));
    }

    // --- Input byte ↔ limb consistency (100 constraints) ---
    for flat_idx in 0..100 {
        add!(KeccakConstraintKind::InputByteLimb(flat_idx));
    }

    // --- Output byte ↔ limb consistency (100 constraints) ---
    for y in 0..5 {
        for x in 0..5 {
            for limb in 0..4 {
                add!(KeccakConstraintKind::OutputByteLimb { x, y, limb });
            }
        }
    }

    // --- Export matches final-step real row (1 constraint) ---
    add!(KeccakConstraintKind::ExportMatchesFinalStep);

    // --- Theta C' xor3 (320 constraints) ---
    for x in 0..5 {
        for z in 0..64 {
            add!(KeccakConstraintKind::ThetaCPrime { x, z });
        }
    }

    // --- Theta parity link (320 constraints) ---
    for x in 0..5 {
        for z in 0..64 {
            add!(KeccakConstraintKind::ThetaParity { x, z });
        }
    }

    // --- A limb from A' (theta link, 100 constraints) ---
    for y in 0..5 {
        for x in 0..5 {
            for limb in 0..4 {
                add!(KeccakConstraintKind::ALimbFromAPrime { x, y, limb });
            }
        }
    }

    // --- Chi limb (100 constraints, degree 3) ---
    for y in 0..5 {
        for x in 0..5 {
            for limb in 0..4 {
                add!(KeccakConstraintKind::ChiLimb { x, y, limb });
            }
        }
    }

    // --- A''[0][0] limb from bits (4 constraints) ---
    for limb in 0..4 {
        add!(KeccakConstraintKind::APP00LimbFromBits(limb));
    }

    // --- Iota limb (4 constraints) ---
    for limb in 0..4 {
        add!(KeccakConstraintKind::IotaLimb(limb));
    }

    // --- Next-round A linkage (100 constraints) ---
    for y in 0..5 {
        for x in 0..5 {
            for limb in 0..4 {
                add!(KeccakConstraintKind::NextAFromOutput { x, y, limb });
            }
        }
    }

    // --- lane_addr = state_addr + 8*lane_idx (50 constraints) ---
    for lane_idx in 0..25 {
        let (c0, c1) = AddConstraint::new_pair(
            vec![cols::FIRST, cols::EXPORT],
            AddOperand::dword(cols::STATE_ADDR_0),
            AddOperand::constant((lane_idx * 8) as i64),
            AddOperand::from_dword_hl(cols::lane_addr(lane_idx)[0]),
            idx,
        );
        constraints.push(Box::new(c0));
        constraints.push(Box::new(c1));
        idx += 2;
    }

    (constraints, idx)
}

/// Create bus interactions for the Keccak chip.
///
/// - `EcallKeccak` receiver on the first row of each real permutation.
/// - 25 `Memw` reads on `first`, binding the preimage to memory at `timestamp`.
/// - 25 `Memw` writes on `export`, binding the final state to memory at `timestamp + 1`.
pub fn bus_interactions() -> Vec<BusInteraction> {
    let mut interactions = Vec::with_capacity(51);
    let syscall_lo = KECCAK_SYSCALL_NUMBER & 0xFFFF_FFFF;
    let syscall_hi = KECCAK_SYSCALL_NUMBER >> 32;

    interactions.push(BusInteraction::receiver(
        BusId::EcallKeccak,
        Multiplicity::Column(cols::FIRST),
        vec![
            BusValue::Packed {
                start_column: cols::TIMESTAMP_0,
                packing: Packing::Direct,
            },
            BusValue::Packed {
                start_column: cols::TIMESTAMP_1,
                packing: Packing::Direct,
            },
            BusValue::constant(syscall_lo),
            BusValue::constant(syscall_hi),
        ],
    ));

    for lane_idx in 0..25 {
        let x = lane_idx % 5;
        let y = lane_idx / 5;
        let input_start = cols::PREIMAGE_BYTES + lane_byte_idx(x, y, 0);
        let output_start = cols::OUTPUT_BYTES + lane_byte_idx(x, y, 0);
        let addr_start = cols::lane_addr(lane_idx)[0];

        let mut read_values = Vec::with_capacity(24);
        for byte in 0..8 {
            read_values.push(BusValue::Packed {
                start_column: input_start + byte,
                packing: Packing::Direct,
            });
        }
        read_values.push(BusValue::constant(0));
        read_values.push(BusValue::Packed {
            start_column: addr_start,
            packing: Packing::DWordHL,
        });
        for byte in 0..8 {
            read_values.push(BusValue::Packed {
                start_column: input_start + byte,
                packing: Packing::Direct,
            });
        }
        read_values.push(BusValue::Packed {
            start_column: cols::TIMESTAMP_0,
            packing: Packing::Direct,
        });
        read_values.push(BusValue::Packed {
            start_column: cols::TIMESTAMP_1,
            packing: Packing::Direct,
        });
        read_values.push(BusValue::constant(0));
        read_values.push(BusValue::constant(0));
        read_values.push(BusValue::constant(1));

        interactions.push(BusInteraction::sender(
            BusId::Memw,
            Multiplicity::Column(cols::FIRST),
            read_values,
        ));

        let mut write_values = Vec::with_capacity(16);
        write_values.push(BusValue::constant(0));
        write_values.push(BusValue::Packed {
            start_column: addr_start,
            packing: Packing::DWordHL,
        });
        for byte in 0..8 {
            write_values.push(BusValue::Packed {
                start_column: output_start + byte,
                packing: Packing::Direct,
            });
        }
        write_values.push(BusValue::linear(vec![
            LinearTerm::Column {
                coefficient: 1,
                column: cols::TIMESTAMP_0,
            },
            LinearTerm::Constant(1),
        ]));
        write_values.push(BusValue::Packed {
            start_column: cols::TIMESTAMP_1,
            packing: Packing::Direct,
        });
        write_values.push(BusValue::constant(0));
        write_values.push(BusValue::constant(0));
        write_values.push(BusValue::constant(1));

        interactions.push(BusInteraction::sender(
            BusId::Memw,
            Multiplicity::Column(cols::EXPORT),
            write_values,
        ));
    }

    interactions
}
