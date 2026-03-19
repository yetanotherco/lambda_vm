//! Keccak-f[1600] permutation chip.
//!
//! A single wide table (2,634 columns) that proves Keccak-f[1600] permutations.
//! Each permutation uses 24 rows (one per round). All intermediate values are
//! stored as columns and verified via purely polynomial constraints — no lookups.
//!
//! ## Design
//!
//! Ported from Plonky3's keccak-air approach: bit-level decomposition of the
//! Keccak state allows expressing theta, rho, pi, chi, and iota as degree-1
//! to degree-3 polynomial constraints.
//!
//! ## Column Layout (2,634 columns)
//!
//! - `step_flags[24]`: One-hot round selector
//! - `export`: Final round flag (= step_flags[23])
//! - `preimage[5][5][4]`: Input state in 16-bit limbs (100 cols)
//! - `a[5][5][4]`: State after theta input (100 cols)
//! - `c[5][64]`: Column parities (320 cols)
//! - `c_prime[5][64]`: Rotated column parities (320 cols)
//! - `a_prime[5][5][64]`: Theta output in bits (1,600 cols)
//! - `a_prime_prime[5][5][4]`: After chi in 16-bit limbs (100 cols)
//! - `a_prime_prime_0_0_bits[64]`: Lane [0,0] bit decomposition (64 cols)
//! - `a_prime_prime_prime_0_0_limbs[4]`: Lane [0,0] after iota (4 cols)
//! - `mu`: Multiplicity flag (1 col)
//!
//! ## Bus Interactions
//!
//! Currently empty — bus interactions will be added incrementally to connect
//! to the ECALL and MEMW buses.
//!
//! ## Constraints
//!
//! Constraints will be added incrementally. The trace generation computes
//! all intermediate values correctly; constraints verify them.

use math::field::element::FieldElement;
use math::field::traits::{IsField, IsSubFieldOf};
use stark::constraints::transition::TransitionConstraint;
use stark::lookup::{BusInteraction, BusValue, LinearTerm, Multiplicity, Packing};
use stark::table::TableView;
use stark::trace::TraceTable;
use stark::traits::TransitionEvaluationContext;

use super::types::{BusId, FE, GoldilocksExtension, GoldilocksField};

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

    /// Export flag: 1 on the final round (round 23) of a real permutation.
    pub const EXPORT: usize = STEP_FLAGS_END;

    /// Preimage: 25 lanes × 4 limbs (16-bit each) = 100 columns.
    /// Layout: preimage[y][x][limb] at offset PREIMAGE + (y*5 + x)*4 + limb.
    pub const PREIMAGE: usize = EXPORT + 1;
    pub const PREIMAGE_END: usize = PREIMAGE + 100;

    /// State A (theta input): same layout as preimage.
    pub const A: usize = PREIMAGE_END;
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

    /// Total number of columns.
    pub const NUM_COLUMNS: usize = STATE_ADDR_1 + 1;
}

// Verify column count at compile time
const _: () = assert!(cols::NUM_COLUMNS == 2638);

// =============================================================================
// Column index helpers
// =============================================================================

/// Index into preimage/a/a_prime_prime arrays: [y][x][limb]
#[inline]
const fn lane_limb_idx(x: usize, y: usize, limb: usize) -> usize {
    (y * 5 + x) * 4 + limb
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
/// Padding rows have mu=0 and valid round flag rotation.
pub fn generate_keccak_trace(
    ops: &[KeccakOperation],
) -> TraceTable<GoldilocksField, GoldilocksExtension> {
    let n_real_rows = ops.len() * 24;
    let num_rows = n_real_rows.next_power_of_two().max(4);
    let mut data = vec![FE::zero(); num_rows * cols::NUM_COLUMNS];

    for (perm_idx, op) in ops.iter().enumerate() {
        let mut state = op.input;

        for round in 0..24 {
            let row_idx = perm_idx * 24 + round;
            let base = row_idx * cols::NUM_COLUMNS;

            // Step flags (one-hot)
            data[base + cols::STEP_FLAGS + round] = FE::one();

            // Export flag (round 23 only)
            if round == 23 {
                data[base + cols::EXPORT] = FE::one();
            }

            // Mu = 1 for real rows
            data[base + cols::MU] = FE::one();

            // Timestamp (same for all 24 rows of this permutation)
            data[base + cols::TIMESTAMP_0] = FE::from(op.timestamp & 0xFFFF_FFFF);
            data[base + cols::TIMESTAMP_1] = FE::from(op.timestamp >> 32);

            // State address (same for all 24 rows)
            data[base + cols::STATE_ADDR_0] = FE::from(op.state_addr & 0xFFFF_FFFF);
            data[base + cols::STATE_ADDR_1] = FE::from(op.state_addr >> 32);

            // Preimage and A: both equal to the current state
            let a = state;
            for y in 0..5 {
                for x in 0..5 {
                    let lane = a[x + 5 * y];
                    for limb in 0..4 {
                        let val = FE::from(limb16(lane, limb));
                        data[base + cols::PREIMAGE + lane_limb_idx(x, y, limb)] = val;
                        data[base + cols::A + lane_limb_idx(x, y, limb)] = val;
                    }
                }
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

            // C'[x][z] = C[(x-1) mod 5][z] XOR C[(x+1) mod 5][(z-1) mod 64]
            let mut c_prime = [[0u64; 64]; 5];
            for x in 0..5 {
                for z in 0..64 {
                    let val = c[(x + 4) % 5][z] ^ c[(x + 1) % 5][(z + 63) % 64];
                    c_prime[x][z] = val;
                    data[base + cols::C_PRIME + parity_idx(x, z)] = FE::from(val);
                }
            }

            // A'[x][y][z] = bit(a[x+5y], z) XOR C'[x][z]
            let mut a_prime = [[[0u64; 64]; 5]; 5];
            for y in 0..5 {
                for x in 0..5 {
                    for z in 0..64 {
                        let val = bit(a[x + 5 * y], z) ^ c_prime[x][z];
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
                        let chi_bit =
                            b[x][y][z] ^ ((1 - b[(x + 1) % 5][y][z]) & b[(x + 2) % 5][y][z]);
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
                data[base + cols::A_PRIME_PRIME_PRIME_0_0_LIMBS + limb] =
                    FE::from(limb16(a_ppp_0_0, limb));
            }

            // Compute output state for this round
            let mut output = a_pp_lanes;
            output[0] = a_ppp_0_0;
            state = output;
        }
    }

    // Fill padding rows: step flags rotate but mu=0 and export=0.
    // export is NOT set on padding rows so that Multiplicity::Column(EXPORT)
    // only fires on real permutation final rounds.
    for row_idx in n_real_rows..num_rows {
        let base = row_idx * cols::NUM_COLUMNS;
        let round = row_idx % 24;
        data[base + cols::STEP_FLAGS + round] = FE::one();
        // export stays 0 on padding (unlike real rows where export=1 on round 23)
        // mu stays 0 (padding), all state columns stay 0

        // Padding iota: A''[0][0]=0, so A'''[0][0] = RC[round]
        let rc = KECCAK_RC[round];
        for limb in 0..4 {
            data[base + cols::A_PRIME_PRIME_PRIME_0_0_LIMBS + limb] = FE::from(limb16(rc, limb));
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
    /// preimage[x][y][limb] = a[x][y][limb]
    PreimageEqualsA(usize), // flat index into the 100 limbs
    /// c'[x][z] = c[(x-1)%5][z] XOR c[(x+1)%5][(z-1)%64]
    ThetaCPrime { x: usize, z: usize },
    /// a''[x][y][limb] = sum of chi bits * 2^k (degree 3)
    ChiLimb { x: usize, y: usize, limb: usize },
    /// a''[0][0][limb] = sum of a_pp_0_0_bits * 2^k
    APP00LimbFromBits(usize),
    /// a'''[0][0][limb] = XOR(a''[0][0], RC[round]) selected by step_flags
    IotaLimb(usize),
    /// a[x][y][limb] = sum of (a'[x][y][z] XOR c'[x][z]) * 2^k
    /// This links A (limbs) to A' (bits) via theta
    ALimbFromAPrime { x: usize, y: usize, limb: usize },
}

struct KeccakConstraint {
    kind: KeccakConstraintKind,
    constraint_idx: usize,
}

impl KeccakConstraint {
    #[allow(clippy::assign_op_pattern)]
    fn compute<F, E>(&self, step: &TableView<F, E>) -> FieldElement<F>
    where
        F: IsSubFieldOf<E>,
        E: IsField,
    {
        let get =
            |col: usize| -> FieldElement<F> { step.get_main_evaluation_element(0, col).clone() };
        let one = FieldElement::<F>::one();
        let two = FieldElement::<F>::from(2u64);

        match self.kind {
            KeccakConstraintKind::PreimageEqualsA(flat_idx) => {
                get(cols::PREIMAGE + flat_idx) - get(cols::A + flat_idx)
            }
            KeccakConstraintKind::ThetaCPrime { x, z } => {
                // c'[x][z] = c[(x-1)%5][z] XOR c[(x+1)%5][(z-1)%64]
                // XOR: a + b - 2ab
                let c_left = get(cols::C + parity_idx((x + 4) % 5, z));
                let c_right = get(cols::C + parity_idx((x + 1) % 5, (z + 63) % 64));
                let cp = get(cols::C_PRIME + parity_idx(x, z));
                let xor_val = c_left.clone() + c_right.clone() - two * c_left * c_right;
                cp - xor_val
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
                    let chi_bit =
                        bxy.clone() + not_b1_and_b2.clone() - two.clone() * bxy * not_b1_and_b2;

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
                        // XOR: a + b - 2ab
                        let xor_bit = a_bit.clone() + rc_bit.clone() - two.clone() * a_bit * rc_bit;
                        let coeff = FieldElement::<F>::from(1u64 << k);
                        round_limb = round_limb + xor_bit * coeff;
                    }
                    iota_limb = iota_limb + flag * round_limb;
                }
                get(cols::A_PRIME_PRIME_PRIME_0_0_LIMBS + limb) - iota_limb
            }
            KeccakConstraintKind::ALimbFromAPrime { x, y, limb } => {
                // a[x][y][limb] = sum_{k=0}^{15} (a'[x][y][16*limb+k] XOR c'[x][16*limb+k]) * 2^k
                // Since a_bit = a' XOR c', and a_bit is the original bit of a[x][y]:
                // XOR: a' + c' - 2*a'*c'
                let mut sum = FieldElement::<F>::zero();
                for k in 0..16 {
                    let z = limb * 16 + k;
                    let ap = get(cols::A_PRIME + a_prime_idx(x, y, z));
                    let cp = get(cols::C_PRIME + parity_idx(x, z));
                    // a_bit = ap XOR cp = ap + cp - 2*ap*cp
                    let a_bit = ap.clone() + cp.clone() - two.clone() * ap * cp;
                    let coeff = FieldElement::<F>::from(1u64 << k);
                    sum = sum + a_bit * coeff;
                }
                get(cols::A + lane_limb_idx(x, y, limb)) - sum
            }
        }
    }
}

impl TransitionConstraint<GoldilocksField, GoldilocksExtension> for KeccakConstraint {
    fn degree(&self) -> usize {
        match self.kind {
            KeccakConstraintKind::PreimageEqualsA(_) => 1,
            KeccakConstraintKind::ThetaCPrime { .. } => 2,
            KeccakConstraintKind::ChiLimb { .. } => 3,
            KeccakConstraintKind::APP00LimbFromBits(_) => 1,
            KeccakConstraintKind::IotaLimb(_) => 2,
            KeccakConstraintKind::ALimbFromAPrime { .. } => 2,
        }
    }

    fn constraint_idx(&self) -> usize {
        self.constraint_idx
    }

    fn end_exemptions(&self) -> usize {
        0
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
                let val = self.compute(frame.get_evaluation_step(0));
                transition_evaluations[self.constraint_idx] = val.to_extension();
            }
            TransitionEvaluationContext::Verifier {
                frame,
                periodic_values: _,
                rap_challenges: _,
                ..
            } => {
                let val = self.compute(frame.get_evaluation_step(0));
                transition_evaluations[self.constraint_idx] = val;
            }
        }
    }
}

/// Create all Keccak constraints. Returns constraints and the next available index.
///
/// Constraint groups:
/// - IS_BIT: step_flags(24), export(1), mu(1), c(320), c'(320), a'(1600), a_pp_0_0_bits(64) = 2,330
/// - Step flag rotation: 24 (transition)
/// - Preimage = A: 100
/// - Preimage consistency: 100 (transition)
/// - Theta C' (XOR): 320
/// - A limb from A' (theta link): 100
/// - Chi limb: 100 (degree 3)
/// - A''[0][0] limb from bits: 4
/// - Iota limb: 4
///
/// Total: ~2,958 (step flag rotation and preimage consistency deferred)
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
    let mut bit_cols: Vec<usize> = Vec::with_capacity(2330);

    // step_flags[0..24]
    for i in 0..24 {
        bit_cols.push(cols::STEP_FLAGS + i);
    }
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

    // NOTE: StepFlagRotation and PreimageConsistency constraints require next-row access,
    // which the current framework doesn't support in TransitionConstraint::compute().
    // These are deferred until the framework supports multi-step constraint evaluation.
    // The trace generation still computes correct values for these relationships.

    // --- Preimage = A (100 constraints) ---
    for flat_idx in 0..100 {
        add!(KeccakConstraintKind::PreimageEqualsA(flat_idx));
    }

    // --- Theta C' XOR (320 constraints) ---
    for x in 0..5 {
        for z in 0..64 {
            add!(KeccakConstraintKind::ThetaCPrime { x, z });
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

    (constraints, idx)
}

// =============================================================================
// Bus interactions (placeholder — will be added incrementally)
// =============================================================================

/// Create bus interactions for the Keccak chip.
///
/// - ECALL receiver: receives [timestamp, rv1] from CPU on the first round
///   of each real permutation (step_flags[0] * mu).
///
/// MEMW interactions for memory reads/writes are handled by the trace builder
/// (via collect_keccak_memw_ops), not by bus interactions on this table.
/// This is the same pattern as the HALT table.
///
/// TODO: Add MEMW bus interactions from the keccak table to bind preimage/output
/// columns to memory values (soundness requirement).
pub fn bus_interactions() -> Vec<BusInteraction> {
    let mut interactions = Vec::new();

    // ECALL receiver: receives from CPU on the first round of each real permutation.
    // Payload: [timestamp_lo, timestamp_hi, syscall_lo, syscall_hi]
    // mult = step_flags[0] * mu (exactly one row per permutation receives)
    let syscall_lo = KECCAK_SYSCALL_NUMBER & 0xFFFF_FFFF;
    let syscall_hi = KECCAK_SYSCALL_NUMBER >> 32;

    interactions.push(BusInteraction::receiver(
        BusId::EcallKeccak,
        Multiplicity::Linear(vec![
            LinearTerm::Column {
                coefficient: 1,
                column: cols::STEP_FLAGS, // step_flags[0]
            },
            // We want step_flags[0] * mu, but Multiplicity::Linear only supports
            // linear combinations, not products. Since step_flags[0] and mu are
            // both 0 or 1, and step_flags[0]=1 implies mu=1 for real rows,
            // we can use just step_flags[0] as the multiplicity.
            // For padding rows: step_flags[0]=1 on row 0 but mu=0.
            // This would incorrectly receive on padding.
            //
            // Fix: use mu alone isn't right either (all 24 rows have mu=1).
            // We need step_flags[0] * mu. Since we can't express products in
            // Multiplicity::Linear, we'll add a dedicated column for this.
            //
            // WORKAROUND: Use step_flags[0] only. On padding rows, step_flags[0]=1
            // on row 0, but the payload (timestamp=0, syscall=constants) won't match
            // any CPU sender, causing bus imbalance only when there are keccak ECALLs.
            // Actually, for programs with no keccak: CPU sends nothing on EcallKeccak,
            // and padding keccak rows receive with step_flags[0] — this IS a problem.
            //
            // Better approach: use mu - (1 - step_flags[0])*mu = step_flags[0]*mu.
            // But that's still a product.
            //
            // Simplest correct solution: For the ECALL, use the EXPORT flag instead.
            // export=1 only on round 23 of real permutations (mu=1 and step_flags[23]=1).
            // On padding: export=1 on the padding round-23 row, but mu=0 there... wait
            // we set export=1 on padding round-23 rows. That's also a problem.
        ]),
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

    // Remove the broken Linear multiplicity and use a proper one.
    // The issue is we need mult = step_flags[0] * mu but can only express linear combos.
    // Solution: Don't set export=1 on padding rows, then use Multiplicity::Column(EXPORT)
    // for the final-round interaction. But we need a first-round interaction for ECALL...
    //
    // Actually, the simplest approach: Use Multiplicity::Column(MU) and receive on
    // a per-permutation basis by having the ECALL interaction fire on EVERY row with mu=1.
    // That would be 24 receives per 1 send — won't balance.
    //
    // Final approach: Add a dedicated FIRST_MU column that is 1 only on the first round
    // of real permutations. OR: just use step_flags[0] and DON'T set step_flags on padding.
    // But step_flags must rotate for constraint correctness...
    //
    // OK let me just fix this properly: modify padding to NOT set export=1, and use
    // export as the multiplicity. Export is only 1 when step_flags[23]=1 AND it's real.

    // Actually, let me clear the broken interaction and do it right.
    interactions.clear();

    // TODO: Add EcallKeccak receiver (mult = export) once MEMW bus interactions
    // are implemented. Without MEMW, enabling this would imbalance Bus 22 since
    // the CPU sender has no matching keccak receiver for the state operations.

    interactions
}
