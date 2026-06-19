//! SHIFT chip for bit-shift operations.
//!
//! Constrains: `out = in <</>>/>>> (shift mod (32 * (2 - word_instr)))`.
//!
//! Two-phase design:
//! 1. Intra-limb shift by `bit_shift = shift mod 16` using paired HWSL lookups (returning [SLL, SLLC]).
//! 2. Full-limb shift by `limb_shift` (unary encoding of `shift >> 4`).
//!
//! ## Columns (26 total)
//! - Input: `in[0..3]` (DWordHL), `shift` (Byte), `direction` (Bit), `signed` (Bit), `word_instr` (Bit)
//! - Output: `out[0..1]` (DWordWL)
//! - Auxiliary: `is_negative`, `bit_shift`, `zbs`, `X[0..4]`, `Y[0..3]`, `limb_shift_raw[0..2]`
//! - Virtual: `limb_shift[3] = 1 - limb_shift_raw[0] - limb_shift_raw[1] - limb_shift_raw[2]`
//! - Multiplicity: `μ`
//!
//! ## Bus Interactions (15 total)
//! - Senders: MSB16, BYTE_ALU[AND] (×3), ZERO, HWSL (×5), IS_HALFWORD (×4)
//! - Receiver: SHIFT (from CPU)

use math::field::element::FieldElement;
use math::field::traits::{IsField, IsSubFieldOf};
use stark::constraints::transition::TransitionConstraint;
use stark::lookup::{BusInteraction, BusValue, LinearTerm, Multiplicity, Packing};
use stark::table::TableView;
use stark::trace::TraceTable;

use super::types::{BusId, FE, GoldilocksExtension, GoldilocksField, SHIFT_16, alu_op};

// =========================================================================
// Column indices
// =========================================================================

pub mod cols {
    // Input: in[0..3] as DWordHL (4 halfwords)
    pub const IN_0: usize = 0;
    pub const IN_1: usize = 1;
    pub const IN_2: usize = 2;
    pub const IN_3: usize = 3;

    // Input: shift amount (byte), direction (bit), signed (bit), word_instr (bit)
    pub const SHIFT_AMOUNT: usize = 4;
    pub const DIRECTION: usize = 5;
    pub const SIGNED: usize = 6;
    pub const WORD_INSTR: usize = 7;

    // Output: out[0..1] as DWordWL (2 words)
    pub const OUT_0: usize = 8;
    pub const OUT_1: usize = 9;

    // Auxiliary
    pub const IS_NEGATIVE: usize = 10;
    pub const BIT_SHIFT: usize = 11;
    pub const ZBS: usize = 12;

    // X[0..4]: 5 half scratch values (intra-limb left shift results)
    pub const X_0: usize = 13;
    pub const X_1: usize = 14;
    pub const X_2: usize = 15;
    pub const X_3: usize = 16;
    pub const X_4: usize = 17;

    // Y[0..3]: 4 half scratch values (intra-limb right shift carry results)
    pub const Y_0: usize = 18;
    pub const Y_1: usize = 19;
    pub const Y_2: usize = 20;
    pub const Y_3: usize = 21;

    // limb_shift_raw[0..2]: first 3 values of the one-hot limb_shift encoding.
    // limb_shift[3] is virtual: 1 - limb_shift_raw[0] - limb_shift_raw[1] - limb_shift_raw[2]
    pub const LIMB_SHIFT_RAW_0: usize = 22;
    pub const LIMB_SHIFT_RAW_1: usize = 23;
    pub const LIMB_SHIFT_RAW_2: usize = 24;

    // Multiplicity
    pub const MU: usize = 25;

    // The unified ALU bus carries the full (un-reduced) shift
    // amount `arg2` as in2. This mirrors the spec's `shift : DWordWHBB` layout
    // `[Byte, Byte, Half, Word]`: SHIFT_AMOUNT (col 4) = shift[0] (low byte, used
    // by the computation, which reduces mod 32/64), then SHIFT_B1 = shift[1],
    // SHIFT_H1 = shift[2], SHIFT_HIGH = shift[3]. The low-word limbs are
    // range-checked (byte/half) so the decomposition is unique → SHIFT_AMOUNT is
    // forced to `arg2 & 0xFF`.
    /// bits 8-15 of the shift amount (byte) — spec `shift[1]`
    pub const SHIFT_B1: usize = 26;
    /// bits 16-31 of the shift amount (half) — spec `shift[2]`
    pub const SHIFT_H1: usize = 27;
    /// bits 32-63 of the shift amount (word) — spec `shift[3]`. `IS_WORD` is
    /// *assumed* (per the spec): on the ALU bus this column equals the CPU's
    /// `arg2` high word, which is already a well-formed 32-bit word, so it needs
    /// no in-chip range check. The high shift bits never affect the result
    /// (`shift mod 32/64` only uses the low byte).
    pub const SHIFT_HIGH: usize = 28;

    pub const NUM_COLUMNS: usize = 29;

    // Helpers for iteration
    pub const IN: [usize; 4] = [IN_0, IN_1, IN_2, IN_3];
    pub const X: [usize; 5] = [X_0, X_1, X_2, X_3, X_4];
    pub const Y: [usize; 4] = [Y_0, Y_1, Y_2, Y_3];
    pub const LIMB_SHIFT_RAW: [usize; 3] = [LIMB_SHIFT_RAW_0, LIMB_SHIFT_RAW_1, LIMB_SHIFT_RAW_2];
}

// =========================================================================
// Trace generation
// =========================================================================

/// A single SHIFT operation (hashable for deduplication).
#[derive(Debug, Clone, Hash, PartialEq, Eq)]
pub struct ShiftOperation {
    /// Input value as 4 halfwords (DWordHL)
    pub in_halves: [u16; 4],
    /// Shift amount low byte (used by the computation; effective = mod 32/64).
    pub shift: u8,
    /// Full shift amount `arg2` (the unified ALU bus carries this as in2).
    pub shift_amount: u64,
    /// 0 = left, 1 = right
    pub direction: bool,
    /// Whether arithmetic (signed) right shift
    pub signed: bool,
    /// Whether 32-bit (word) instruction
    pub word_instr: bool,
}

impl ShiftOperation {
    /// `shift_amount` is the full (un-reduced) shift operand `arg2`; only its low
    /// byte feeds the computation (the result depends on `arg2 mod 32/64`).
    pub fn new(
        value: u64,
        shift_amount: u64,
        direction: bool,
        signed: bool,
        word_instr: bool,
    ) -> Self {
        Self {
            in_halves: [
                (value & 0xFFFF) as u16,
                ((value >> 16) & 0xFFFF) as u16,
                ((value >> 32) & 0xFFFF) as u16,
                ((value >> 48) & 0xFFFF) as u16,
            ],
            shift: (shift_amount & 0xFF) as u8,
            shift_amount,
            direction,
            signed,
            word_instr,
        }
    }

    /// Compute HWSL: (halfword << z) & 0xFFFF
    fn hwsl(halfword: u16, z: u8) -> u16 {
        if z == 0 {
            halfword
        } else {
            ((halfword as u32) << z as u32) as u16
        }
    }

    /// Compute the carry output of HWSL: halfword >> (16 - z)
    /// This is the second element of the HWSL pair [SLL, SLLC].
    fn hwslc(halfword: u16, z: u8) -> u16 {
        if z == 0 {
            0
        } else {
            halfword >> (16 - z as u16)
        }
    }

    /// Compute the full result.
    pub fn compute_result(&self) -> u64 {
        let val = self.in_halves[0] as u64
            | (self.in_halves[1] as u64) << 16
            | (self.in_halves[2] as u64) << 32
            | (self.in_halves[3] as u64) << 48;

        let modulus = if self.word_instr { 32 } else { 64 };
        let effective_shift = (self.shift as u32) % modulus;

        if !self.direction {
            // Left shift
            if self.word_instr {
                // 32-bit: shift lower 32 bits, sign-extend
                let lo = (val as u32).wrapping_shl(effective_shift);
                lo as i32 as i64 as u64
            } else {
                val.wrapping_shl(effective_shift)
            }
        } else if !self.signed {
            // Logical right shift
            if self.word_instr {
                let lo = (val as u32).wrapping_shr(effective_shift);
                lo as i32 as i64 as u64
            } else {
                val.wrapping_shr(effective_shift)
            }
        } else {
            // Arithmetic right shift
            if self.word_instr {
                let lo = (val as i32).wrapping_shr(effective_shift);
                lo as i64 as u64
            } else {
                (val as i64).wrapping_shr(effective_shift) as u64
            }
        }
    }

    /// The raw shift output the chip writes to `OUT` (DWordWL) and sends on the
    /// ALU bus as `res`. Unlike [`compute_result`](Self::compute_result), this is
    /// NOT sign-extended for word shifts — the CPU32 applies that extension to
    /// obtain `rvd`. For non-word shifts the two coincide.
    pub fn compute_out(&self) -> u64 {
        let aux = self.compute_aux();
        aux.out[0] as u64 | ((aux.out[1] as u64) << 32)
    }

    /// Compute all auxiliary values for trace generation.
    fn compute_aux(&self) -> ShiftAux {
        let left = !self.direction;
        let right = self.direction;

        // is_negative is the MSB of in[3] BUT gated by `signed`. The SHIFT
        // AIR constrains IS_NEGATIVE via the MSB16 bus (SHIFT-C14) only when
        // `signed = 1` — for `signed = 0` IS_NEGATIVE is free, so we set it
        // to zero. This makes `extension = 65535 * is_negative = 0` for SRL,
        // so the extension contribution in `compute_shifted_half` naturally
        // vanishes (zero fill) — matching RISC-V SRL semantics regardless of
        // the top-bit value of the input.
        let is_negative = self.signed && (self.in_halves[3] >> 15) & 1 == 1;
        let extension: u16 = if is_negative { 0xFFFF } else { 0 };

        // bit_shift
        let bit_shift = if left {
            self.shift & 15
        } else {
            (256u16.wrapping_sub(self.shift as u16) & 15) as u8
        };

        let zbs = bit_shift == 0;

        // X[0..3] and Y[0..3]
        let mut x = [0u16; 5];
        let mut y = [0u16; 4];

        if zbs {
            // Override when bit_shift == 0
            for i in 0..4 {
                if left {
                    x[i] = self.in_halves[i];
                } else {
                    y[i] = self.in_halves[i];
                }
            }
            x[4] = 0;
        } else {
            for i in 0..4 {
                x[i] = Self::hwsl(self.in_halves[i], bit_shift);
                y[i] = Self::hwslc(self.in_halves[i], bit_shift);
            }
            x[4] = Self::hwsl(extension, bit_shift);
        }

        // limb_shift: unary encoding of (shift >> 4) & mask
        let limb_idx = if self.word_instr {
            ((self.shift >> 4) & 1) as usize
        } else {
            ((self.shift >> 4) & 3) as usize
        };
        let mut limb_shift = [false; 4];
        limb_shift[limb_idx] = true;

        // Compute shifted as DWordHL (4 halfwords)
        let shifted = self.compute_shifted(&x, &y, extension, &limb_shift, left, right);

        // Cast shifted to DWordWL (spec C14: out[i] = (shifted::DWordWL)[i])
        let out_0 = shifted[0] as u32 | (shifted[1] as u32) << 16;
        let out_1 = shifted[2] as u32 | (shifted[3] as u32) << 16;

        ShiftAux {
            is_negative,
            bit_shift,
            zbs,
            x,
            y,
            limb_shift,
            out: [out_0, out_1],
        }
    }

    /// Compute the `shifted` virtual column (DWordHL, 4 halfwords).
    fn compute_shifted(
        &self,
        x: &[u16; 5],
        y: &[u16; 4],
        extension: u16,
        limb_shift: &[bool; 4],
        left: bool,
        right: bool,
    ) -> [u16; 4] {
        // intra_limb_left[i] = X[0] for i=0, X[i]+Y[i-1] for i>0
        let intra_left = |i: usize| -> u16 {
            if i == 0 {
                x[0]
            } else {
                x[i].wrapping_add(y[i - 1])
            }
        };

        // intra_limb_right[i] = Y[i]+X[i+1]
        let intra_right = |i: usize| -> u16 { y[i].wrapping_add(x[i + 1]) };

        let mut shifted = [0u16; 4];
        for (i, shifted_i) in shifted.iter_mut().enumerate() {
            let mut val = 0u16;

            if left {
                // left * Σ_j=0^i limb_shift[j] * intra_limb_left[i-j]
                for (j, &ls_j) in limb_shift.iter().enumerate().take(i + 1) {
                    if ls_j {
                        val = val.wrapping_add(intra_left(i - j));
                    }
                }
            }

            if right {
                // right * (Σ_j=0^(3-i) limb_shift[j] * intra_limb_right[i+j]
                //          + extension * Σ_j=(3-i+1)^3 limb_shift[j])
                for (j, &ls_j) in limb_shift.iter().enumerate().take(3 - i + 1) {
                    if ls_j {
                        val = val.wrapping_add(intra_right(i + j));
                    }
                }
                for &ls_j in limb_shift.iter().take(4).skip(4 - i) {
                    if ls_j {
                        val = val.wrapping_add(extension);
                    }
                }
            }

            *shifted_i = val;
        }
        shifted
    }
}

struct ShiftAux {
    is_negative: bool,
    bit_shift: u8,
    zbs: bool,
    x: [u16; 5],
    y: [u16; 4],
    limb_shift: [bool; 4],
    out: [u32; 2],
}

/// Generates the SHIFT trace table from a list of operations.
pub fn generate_shift_trace(
    operations: &[ShiftOperation],
) -> TraceTable<GoldilocksField, GoldilocksExtension> {
    // No deduplication: each operation gets its own row with μ=1.
    // Spec declares μ: Bit.
    let num_rows = operations.len().next_power_of_two().max(4);

    // GPU fast path: pack each op into three parallel u64 arrays
    // (value, shift_amount, flags) and let the kernel do compute_aux +
    // row layout. Padding rows get only ZBS=1 on-device.
    #[cfg(feature = "cuda")]
    if let Some(table) = try_generate_shift_trace_gpu(operations, num_rows) {
        return table;
    }

    let mut data = vec![FE::zero(); num_rows * cols::NUM_COLUMNS];

    for (row_idx, op) in operations.iter().enumerate() {
        let base = row_idx * cols::NUM_COLUMNS;
        let aux = op.compute_aux();

        // Input columns
        for i in 0..4 {
            data[base + cols::IN[i]] = FE::from(op.in_halves[i] as u64);
        }
        data[base + cols::SHIFT_AMOUNT] = FE::from(op.shift as u64);
        // High bits of the full shift amount (for the ALU bus in2 = arg2).
        data[base + cols::SHIFT_B1] = FE::from((op.shift_amount >> 8) & 0xFF);
        data[base + cols::SHIFT_H1] = FE::from((op.shift_amount >> 16) & 0xFFFF);
        data[base + cols::SHIFT_HIGH] = FE::from(op.shift_amount >> 32);
        data[base + cols::DIRECTION] = FE::from(op.direction as u64);
        data[base + cols::SIGNED] = FE::from(op.signed as u64);
        data[base + cols::WORD_INSTR] = FE::from(op.word_instr as u64);

        // Output columns
        data[base + cols::OUT_0] = FE::from(aux.out[0] as u64);
        data[base + cols::OUT_1] = FE::from(aux.out[1] as u64);

        // Auxiliary columns
        data[base + cols::IS_NEGATIVE] = FE::from(aux.is_negative as u64);
        data[base + cols::BIT_SHIFT] = FE::from(aux.bit_shift as u64);
        data[base + cols::ZBS] = FE::from(aux.zbs as u64);

        for i in 0..5 {
            data[base + cols::X[i]] = FE::from(aux.x[i] as u64);
        }
        for i in 0..4 {
            data[base + cols::Y[i]] = FE::from(aux.y[i] as u64);
        }
        for i in 0..3 {
            data[base + cols::LIMB_SHIFT_RAW[i]] = FE::from(aux.limb_shift[i] as u64);
        }
        // limb_shift[3] is virtual: not stored in the trace

        // μ = 1 for all active rows (Bit)
        data[base + cols::MU] = FE::one();
    }

    // Padding rows: set ZBS=1 per spec. All other columns remain 0.
    // μ=0 so C13 (limb_shift encoding) is inactive. left=right=0 so shifted=0,
    // making C14 (out=shifted) trivially satisfied regardless of limb_shift.
    for row_idx in operations.len()..num_rows {
        let base = row_idx * cols::NUM_COLUMNS;
        data[base + cols::ZBS] = FE::one();
    }

    TraceTable::new_main(data, cols::NUM_COLUMNS, 1)
}

/// CUDA fast path for `generate_shift_trace`. Packs each op into three
/// parallel u64 arrays (length `num_rows`, padded). Active-row flags use
/// bits 0..3 = direction | signed | word_instr | mu(=1); inactive rows
/// keep `flags = 0` so the kernel writes only ZBS=1 there.
#[cfg(feature = "cuda")]
fn try_generate_shift_trace_gpu(
    operations: &[ShiftOperation],
    num_rows: usize,
) -> Option<TraceTable<GoldilocksField, GoldilocksExtension>> {
    let mut in_values = vec![0u64; num_rows];
    let mut shift_amounts = vec![0u64; num_rows];
    let mut flags = vec![0u64; num_rows];
    for (i, op) in operations.iter().enumerate() {
        in_values[i] = (op.in_halves[0] as u64)
            | ((op.in_halves[1] as u64) << 16)
            | ((op.in_halves[2] as u64) << 32)
            | ((op.in_halves[3] as u64) << 48);
        shift_amounts[i] = op.shift_amount;
        let mut f: u64 = 0;
        if op.direction {
            f |= 1;
        }
        if op.signed {
            f |= 1 << 1;
        }
        if op.word_instr {
            f |= 1 << 2;
        }
        f |= 1 << 3; // active (mu = 1)
        flags[i] = f;
    }

    let raw = stark::gpu_lde::try_generate_shift_trace_gpu_raw(
        num_rows,
        &in_values,
        &shift_amounts,
        &flags,
        cols::NUM_COLUMNS,
    )?;
    let data: Vec<FE> = raw.into_iter().map(FE::from).collect();
    Some(TraceTable::new_main(data, cols::NUM_COLUMNS, 1))
}

// =========================================================================
// Bus interactions
// =========================================================================

/// Creates all bus interactions for the SHIFT table.
pub fn bus_interactions() -> Vec<BusInteraction> {
    let mut interactions = Vec::with_capacity(15);

    // SHIFT-C14: MSB16[in[3]] → is_negative | signed
    interactions.push(BusInteraction::sender(
        BusId::Msb16,
        Multiplicity::Column(cols::SIGNED),
        vec![
            // in[3] as halfword: x + 256*y (in[3] is stored as single Half column)
            BusValue::Packed {
                start_column: cols::IN_3,
                packing: Packing::Direct,
            },
            BusValue::Packed {
                start_column: cols::IS_NEGATIVE,
                packing: Packing::Direct,
            },
        ],
    ));

    // SHIFT-C1: BYTE_ALU[bit_shift; AND, shift, 15] | left (= μ - direction)
    interactions.push(BusInteraction::sender(
        BusId::ByteAlu,
        Multiplicity::Diff(cols::MU, cols::DIRECTION),
        vec![
            BusValue::constant(alu_op::AND as u64),
            BusValue::Packed {
                start_column: cols::SHIFT_AMOUNT,
                packing: Packing::Direct,
            },
            BusValue::constant(15),
            BusValue::Packed {
                start_column: cols::BIT_SHIFT,
                packing: Packing::Direct,
            },
        ],
    ));

    // SHIFT-C2: BYTE_ALU[bit_shift; AND, 256 - zbs * 16 - shift, 15] | right
    // (= direction)
    // 256 - shift would overflow a byte when shift = 0. Subtracting zbs * 16 keeps it in
    // [0,255].
    // When zbs = 1, shift is a multiple of 16 (i.e. shift ∈ [0, 240]), so
    // 256 - 16 - shift ∈ [0,255].
    interactions.push(BusInteraction::sender(
        BusId::ByteAlu,
        Multiplicity::Column(cols::DIRECTION),
        vec![
            BusValue::constant(alu_op::AND as u64),
            BusValue::linear(vec![
                LinearTerm::Constant(256),
                LinearTerm::Column {
                    coefficient: -16,
                    column: cols::ZBS,
                },
                LinearTerm::Column {
                    coefficient: -1,
                    column: cols::SHIFT_AMOUNT,
                },
            ]),
            BusValue::constant(15),
            BusValue::Packed {
                start_column: cols::BIT_SHIFT,
                packing: Packing::Direct,
            },
        ],
    ));

    // SHIFT-C3: ZERO[bit_shift] → zbs | μ
    // ZERO receiver expects [x + 256*y + 65536*z, zero_flag]
    // bit_shift is a byte (0-15), so y=0, z=0: just send bit_shift directly
    interactions.push(BusInteraction::sender(
        BusId::Zero,
        Multiplicity::Column(cols::MU),
        vec![
            BusValue::Packed {
                start_column: cols::BIT_SHIFT,
                packing: Packing::Direct,
            },
            BusValue::Packed {
                start_column: cols::ZBS,
                packing: Packing::Direct,
            },
        ],
    ));

    // SHIFT-C4.i: HWSL[in[i], bit_shift] → [X[i], Y[i]] for i∈[0,3] | 1 - zbs
    // HWSL receiver: [x + 256*y (halfword), z (shift amount), SLL, SLLC]
    let one_minus_zbs = Multiplicity::Negated(cols::ZBS);
    for i in 0..4 {
        interactions.push(BusInteraction::sender(
            BusId::Hwsl,
            one_minus_zbs.clone(),
            vec![
                BusValue::Packed {
                    start_column: cols::IN[i],
                    packing: Packing::Direct,
                },
                BusValue::Packed {
                    start_column: cols::BIT_SHIFT,
                    packing: Packing::Direct,
                },
                BusValue::Packed {
                    start_column: cols::X[i],
                    packing: Packing::Direct,
                },
                BusValue::Packed {
                    start_column: cols::Y[i],
                    packing: Packing::Direct,
                },
            ],
        ));
    }

    // SHIFT-C7: HWSL[extension, bit_shift] → [X[4], extension - X[4]] | 1 - zbs
    // extension = 65535 * is_negative (virtual)
    // second output = extension - X[4] (the carry, expressed as a linear combination)
    interactions.push(BusInteraction::sender(
        BusId::Hwsl,
        one_minus_zbs.clone(),
        vec![
            BusValue::linear(vec![LinearTerm::Column {
                coefficient: 65535,
                column: cols::IS_NEGATIVE,
            }]),
            BusValue::Packed {
                start_column: cols::BIT_SHIFT,
                packing: Packing::Direct,
            },
            BusValue::Packed {
                start_column: cols::X_4,
                packing: Packing::Direct,
            },
            BusValue::linear(vec![
                LinearTerm::Column {
                    coefficient: 65535,
                    column: cols::IS_NEGATIVE,
                },
                LinearTerm::Column {
                    coefficient: -1,
                    column: cols::X_4,
                },
            ]),
        ],
    ));

    // SHIFT-C11: BYTE_ALU[encoded_limb; AND, shift, mask] | μ
    // encoded = (1 - ls[0]) + 15*ls[1] + 31*ls[2] + 47*ls[3]
    // mask = 48 - 32 * word_instr
    interactions.push(BusInteraction::sender(
        BusId::ByteAlu,
        Multiplicity::Column(cols::MU),
        vec![
            BusValue::constant(alu_op::AND as u64),
            // first input: shift
            BusValue::Packed {
                start_column: cols::SHIFT_AMOUNT,
                packing: Packing::Direct,
            },
            // second input: mask = 48 - 32 * word_instr
            BusValue::linear(vec![
                LinearTerm::Constant(48),
                LinearTerm::Column {
                    coefficient: -32,
                    column: cols::WORD_INSTR,
                },
            ]),
            // result: encoded limb_shift
            // = (1 - ls[0]) + 15*ls[1] + 31*ls[2] + 47*ls[3]
            // substituting ls[3] = 1 - ls_raw[0] - ls_raw[1] - ls_raw[2]:
            // = 48 - 48*ls_raw[0] - 32*ls_raw[1] - 16*ls_raw[2]
            BusValue::linear(vec![
                LinearTerm::Constant(48),
                LinearTerm::Column {
                    coefficient: -48,
                    column: cols::LIMB_SHIFT_RAW_0,
                },
                LinearTerm::Column {
                    coefficient: -32,
                    column: cols::LIMB_SHIFT_RAW_1,
                },
                LinearTerm::Column {
                    coefficient: -16,
                    column: cols::LIMB_SHIFT_RAW_2,
                },
            ]),
        ],
    ));

    // Unified ALU receiver: the CPU dispatches SLL/SRL/SRA here.
    // ALU[out::DWordWL; in1=in, in2=shift_amount, flags] where
    //   flags = opsel(SHIFT=5, +word_instr→SHIFTW=6) + 32*signed + 64*direction.
    // in2 = the full shift amount: [SHIFT_AMOUNT + 256*SHIFT_B1 + 2^16*SHIFT_H1,
    //                               SHIFT_HIGH].
    interactions.push(BusInteraction::receiver(
        BusId::Alu,
        Multiplicity::Column(cols::MU),
        vec![
            // in1 = in as DWordHL (4 halfwords → 2 words)
            BusValue::Packed {
                start_column: cols::IN_0,
                packing: Packing::DWordHL,
            },
            // in2 = full shift amount, low word
            BusValue::linear(vec![
                LinearTerm::Column {
                    coefficient: 1,
                    column: cols::SHIFT_AMOUNT,
                },
                LinearTerm::Column {
                    coefficient: 1 << 8,
                    column: cols::SHIFT_B1,
                },
                LinearTerm::Column {
                    coefficient: 1 << 16,
                    column: cols::SHIFT_H1,
                },
            ]),
            // in2 high word = arg2 bits 32-63 (spec `shift[3]`, a Word; IS_WORD
            // assumed via this column's bus equality with the CPU's well-formed
            // arg2 high word).
            BusValue::Packed {
                start_column: cols::SHIFT_HIGH,
                packing: Packing::Direct,
            },
            // flags = opsel(SHIFT) + word_instr + 32*signed + 64*direction
            BusValue::linear(vec![
                LinearTerm::Constant(alu_op::SHIFT as i64),
                LinearTerm::Column {
                    coefficient: 1,
                    column: cols::WORD_INSTR,
                },
                LinearTerm::Column {
                    coefficient: 32,
                    column: cols::SIGNED,
                },
                LinearTerm::Column {
                    coefficient: 64,
                    column: cols::DIRECTION,
                },
            ]),
            // out as DWordWL (2 elements)
            BusValue::Packed {
                start_column: cols::OUT_0,
                packing: Packing::DWordWL,
            },
        ],
    ));

    // Range checks for the low-word high bits (so the in2 low-word decomposition
    // is unique → SHIFT_AMOUNT is forced to `arg2 & 0xFF`). SHIFT_AMOUNT is also
    // byte-checked implicitly via the BYTE_ALU[AND, shift, mask] lookups; we still emit
    // the explicit ARE_BYTES[shift[0]] below to match the spec's `IS_BYTE[shift[0]]`
    // (defense-in-depth, redundant with BYTE_ALU[AND]). SHIFT_HIGH (the high word) needs
    // no check: IS_WORD is assumed (it equals the CPU's well-formed arg2 high word
    // on the bus), matching the spec's `shift[3]`.
    interactions.push(BusInteraction::sender(
        BusId::AreBytes,
        Multiplicity::Column(cols::MU),
        vec![
            BusValue::Packed {
                start_column: cols::SHIFT_B1,
                packing: Packing::Direct,
            },
            BusValue::constant(0),
        ],
    ));
    interactions.push(BusInteraction::sender(
        BusId::AreBytes,
        Multiplicity::Column(cols::MU),
        vec![
            BusValue::Packed {
                start_column: cols::SHIFT_AMOUNT,
                packing: Packing::Direct,
            },
            BusValue::constant(0),
        ],
    ));
    interactions.push(BusInteraction::sender(
        BusId::IsHalfword,
        Multiplicity::Column(cols::MU),
        vec![BusValue::Packed {
            start_column: cols::SHIFT_H1,
            packing: Packing::Direct,
        }],
    ));

    // VM-3: range-check every input half `in[i]` as a 16-bit value, unconditionally
    // on every active row. The SHIFT bus carries only the *packed* operand, so
    // without these a non-canonical half-decomposition that wraps in the field
    // (keeping the packed word constant) would be invisible to the caller while
    // still changing the shifted output.
    for input_col in cols::IN {
        interactions.push(BusInteraction::sender(
            BusId::IsHalfword,
            Multiplicity::Column(cols::MU),
            vec![BusValue::Packed {
                start_column: input_col,
                packing: Packing::Direct,
            }],
        ));
    }

    interactions
}

// =========================================================================
// Constraints
// =========================================================================

/// Polynomial constraint kinds for the SHIFT table.
#[derive(Debug, Clone, Copy)]
pub enum ShiftConstraintKind {
    /// SHIFT-C13: direction * (1 - μ) = 0
    DirectionImpliesMu,
    /// SHIFT-C5.i: zbs * (X[i] - in[i] * left) = 0
    ZbsOverrideX(usize),
    /// SHIFT-C7: zbs * X[4] = 0
    ZbsOverrideX4,
    /// SHIFT-C9.i: zbs * (Y[i] - in[i] * right) = 0
    ZbsOverrideY(usize),
    /// SHIFT-C10.i: IS_BIT<limb_shift[i]>
    LimbShiftIsBit(usize),
    /// SHIFT-C12.i: out[i] - (shifted::DWordWL)[i] = 0
    OutputMatchesShifted(usize),
    /// `IS_BIT<flag>`: `flag * (1 - flag) = 0` for a boolean flag used as a bus
    /// multiplicity / shift selector (`shift:c:direction|signed|word_instr`).
    /// `usize` is the flag column.
    FlagIsBit(usize),
}

pub struct ShiftConstraint {
    constraint_idx: usize,
    kind: ShiftConstraintKind,
}

impl ShiftConstraint {
    pub fn new(kind: ShiftConstraintKind, constraint_idx: usize) -> Self {
        Self {
            constraint_idx,
            kind,
        }
    }

    /// Compute the `shifted` virtual column at index `half_idx` (0..4).
    fn compute_shifted_half<F, E>(half_idx: usize, step: &TableView<F, E>) -> FieldElement<F>
    where
        F: IsSubFieldOf<E>,
        E: IsField,
    {
        let dir: FieldElement<F> = step.get_main_evaluation_element(0, cols::DIRECTION).clone();
        let mu = step.get_main_evaluation_element(0, cols::MU).clone();
        let left = &mu - &dir; // μ - direction
        let right = dir;

        // extension = 65535 * is_negative
        let is_neg = step.get_main_evaluation_element(0, cols::IS_NEGATIVE);
        let extension = is_neg * FieldElement::<F>::from(65535u64);

        // Get X, Y, limb_shift, in columns
        let get_x = |i: usize| step.get_main_evaluation_element(0, cols::X[i]).clone();
        let get_y = |i: usize| step.get_main_evaluation_element(0, cols::Y[i]).clone();
        let get_ls = |i: usize| -> FieldElement<F> {
            if i < 3 {
                step.get_main_evaluation_element(0, cols::LIMB_SHIFT_RAW[i])
                    .clone()
            } else {
                // limb_shift[3] is virtual: 1 - ls_raw[0] - ls_raw[1] - ls_raw[2]
                FieldElement::<F>::one()
                    - step.get_main_evaluation_element(0, cols::LIMB_SHIFT_RAW[0])
                    - step.get_main_evaluation_element(0, cols::LIMB_SHIFT_RAW[1])
                    - step.get_main_evaluation_element(0, cols::LIMB_SHIFT_RAW[2])
            }
        };

        // intra_limb_left[i]: X[0] for i=0, X[i]+Y[i-1] for i>0
        let intra_left = |i: usize| -> FieldElement<F> {
            if i == 0 {
                get_x(0)
            } else {
                get_x(i) + get_y(i - 1)
            }
        };

        // intra_limb_right[i]: Y[i]+X[i+1]
        let intra_right = |i: usize| -> FieldElement<F> { get_y(i) + get_x(i + 1) };

        let i = half_idx;
        let zero = FieldElement::<F>::zero();

        // left_part = left * Σ_j=0^i limb_shift[j] * intra_limb_left[i-j]
        let mut left_part = zero.clone();
        for j in 0..=i {
            left_part += &get_ls(j) * intra_left(i - j);
        }
        left_part = &left * left_part;

        // right_shift_part = right * Σ_j=0^(3-i) limb_shift[j] * intra_limb_right[i+j]
        let mut right_shift_part = zero.clone();
        for j in 0..=(3 - i) {
            right_shift_part += &get_ls(j) * intra_right(i + j);
        }

        // right_ext_part = right * extension * Σ_j=(4-i)^3 limb_shift[j]
        let mut ext_sum = zero.clone();
        if i < 4 {
            for j in (4 - i)..4 {
                ext_sum += get_ls(j);
            }
        }
        let right_ext_part = &extension * ext_sum;

        let right_part = &right * (right_shift_part + right_ext_part);

        left_part + right_part
    }

    fn compute<F, E>(&self, step: &TableView<F, E>) -> FieldElement<F>
    where
        F: IsSubFieldOf<E>,
        E: IsField,
    {
        let one = FieldElement::<F>::one();
        let shift_16 = FieldElement::<F>::from(SHIFT_16);

        match self.kind {
            ShiftConstraintKind::DirectionImpliesMu => {
                // direction * (1 - μ) = 0
                let dir = step.get_main_evaluation_element(0, cols::DIRECTION);
                let mu = step.get_main_evaluation_element(0, cols::MU);
                dir * (&one - mu)
            }
            ShiftConstraintKind::ZbsOverrideX(i) => {
                // zbs * (X[i] - in[i] * left) = 0, where left = μ - direction
                let zbs = step.get_main_evaluation_element(0, cols::ZBS);
                let x_i = step.get_main_evaluation_element(0, cols::X[i]);
                let in_i = step.get_main_evaluation_element(0, cols::IN[i]);
                let mu = step.get_main_evaluation_element(0, cols::MU);
                let dir = step.get_main_evaluation_element(0, cols::DIRECTION);
                let left = mu - dir;
                zbs * (x_i - in_i * &left)
            }
            ShiftConstraintKind::ZbsOverrideX4 => {
                // zbs * X[4] = 0
                let zbs = step.get_main_evaluation_element(0, cols::ZBS);
                let x4 = step.get_main_evaluation_element(0, cols::X_4);
                zbs * x4
            }
            ShiftConstraintKind::ZbsOverrideY(i) => {
                // zbs * (Y[i] - in[i] * right) = 0
                let zbs = step.get_main_evaluation_element(0, cols::ZBS);
                let y_i = step.get_main_evaluation_element(0, cols::Y[i]);
                let in_i = step.get_main_evaluation_element(0, cols::IN[i]);
                let dir = step.get_main_evaluation_element(0, cols::DIRECTION);
                zbs * (y_i - in_i * dir)
            }
            ShiftConstraintKind::LimbShiftIsBit(i) => {
                // limb_shift[i] * (1 - limb_shift[i]) = 0
                // limb_shift[3] is virtual: 1 - ls_raw[0] - ls_raw[1] - ls_raw[2]
                let ls = if i < 3 {
                    step.get_main_evaluation_element(0, cols::LIMB_SHIFT_RAW[i])
                        .clone()
                } else {
                    one.clone()
                        - step.get_main_evaluation_element(0, cols::LIMB_SHIFT_RAW[0])
                        - step.get_main_evaluation_element(0, cols::LIMB_SHIFT_RAW[1])
                        - step.get_main_evaluation_element(0, cols::LIMB_SHIFT_RAW[2])
                };
                &ls * (&one - &ls)
            }
            ShiftConstraintKind::OutputMatchesShifted(i) => {
                // C12.i: out[i] - (shifted::DWordWL)[i] = 0
                // (shifted::DWordWL)[i] = shifted[2*i] + shifted[2*i+1] * 2^16
                let out_col = if i == 0 { cols::OUT_0 } else { cols::OUT_1 };
                let out = step.get_main_evaluation_element(0, out_col).clone();
                let half_lo = Self::compute_shifted_half(2 * i, step);
                let half_hi = Self::compute_shifted_half(2 * i + 1, step);
                out - half_lo - half_hi * shift_16
            }
            ShiftConstraintKind::FlagIsBit(col) => {
                // flag * (1 - flag) = 0
                let flag = step.get_main_evaluation_element(0, col).clone();
                let one = FieldElement::<F>::one();
                &flag * (one - &flag)
            }
        }
    }
}

impl TransitionConstraint<GoldilocksField, GoldilocksExtension> for ShiftConstraint {
    fn degree(&self) -> usize {
        match self.kind {
            ShiftConstraintKind::DirectionImpliesMu => 2,
            ShiftConstraintKind::ZbsOverrideX(_) => 3, // zbs * (X - in * left), left = 1 - dir
            ShiftConstraintKind::ZbsOverrideX4 => 2,
            ShiftConstraintKind::ZbsOverrideY(_) => 3, // zbs * (Y - in * dir)
            ShiftConstraintKind::LimbShiftIsBit(_) => 2,
            ShiftConstraintKind::OutputMatchesShifted(_) => 3, // out - left*ls*intra (degree 3)
            ShiftConstraintKind::FlagIsBit(_) => 2,
        }
    }

    fn constraint_idx(&self) -> usize {
        self.constraint_idx
    }

    fn evaluate<F, E>(&self, step: &TableView<F, E>) -> FieldElement<F>
    where
        F: IsSubFieldOf<E>,
        E: IsField,
    {
        self.compute(step)
    }
}

/// Number of polynomial constraints in the SHIFT table.
// 1 (DirectionImpliesMu) + 4 (ZbsOverrideX) + 1 (ZbsOverrideX4) + 4 (ZbsOverrideY)
// + 4 (LimbShiftIsBit) + 2 (OutputMatchesShifted) + 3 (FlagIsBit) = 19
pub const NUM_SHIFT_CONSTRAINTS: usize = 19;

/// Creates all polynomial constraints for the SHIFT table.
pub fn shift_constraints(constraint_idx_start: usize) -> (Vec<ShiftConstraint>, usize) {
    let mut idx = constraint_idx_start;
    let mut constraints = Vec::with_capacity(NUM_SHIFT_CONSTRAINTS);

    let mut push = |kind| {
        constraints.push(ShiftConstraint::new(kind, idx));
        idx += 1;
    };

    // C13: direction * (1 - μ) = 0
    push(ShiftConstraintKind::DirectionImpliesMu);

    // C5.i: zbs * (X[i] - in[i] * left) = 0
    for i in 0..4 {
        push(ShiftConstraintKind::ZbsOverrideX(i));
    }

    // C7: zbs * X[4] = 0
    push(ShiftConstraintKind::ZbsOverrideX4);

    // C9.i: zbs * (Y[i] - in[i] * right) = 0
    for i in 0..4 {
        push(ShiftConstraintKind::ZbsOverrideY(i));
    }

    // C10.i: IS_BIT<limb_shift[i]>
    for i in 0..4 {
        push(ShiftConstraintKind::LimbShiftIsBit(i));
    }

    // C12.i: out[i] - (shifted::DWordWL)[i] = 0
    for i in 0..2 {
        push(ShiftConstraintKind::OutputMatchesShifted(i));
    }

    // IS_BIT[direction|signed|word_instr] (shift.toml `range` group): these flags
    // drive bus multiplicities / shift selectors, so they must be boolean.
    for flag_col in [cols::DIRECTION, cols::SIGNED, cols::WORD_INSTR] {
        push(ShiftConstraintKind::FlagIsBit(flag_col));
    }

    debug_assert_eq!(constraints.len(), NUM_SHIFT_CONSTRAINTS);
    (constraints, idx)
}

// =========================================================================
// Bitwise operation collection
// =========================================================================

use super::bitwise::{BitwiseOperation, BitwiseOperationType};

/// Collect BITWISE table lookups needed by a set of unique shift operations.
///
/// Each unique operation (with its multiplicity) generates HWSL/BYTE_ALU/MSB16/ZERO
/// lookups. The lookups must be generated per-unique-operation (matching the SHIFT table's
/// deduplication and μ column), and repeated `multiplicity` times.
pub fn collect_bitwise_from_shift(operations: &[ShiftOperation]) -> Vec<BitwiseOperation> {
    // No deduplication: each operation has μ=1, matching generate_shift_trace.
    let mut bitwise_ops = Vec::new();

    for op in operations {
        let aux = op.compute_aux();
        let left = !op.direction;
        let right = op.direction;

        // C14: MSB16[in[3]] | signed
        if op.signed {
            let x = (op.in_halves[3] & 0xFF) as u8;
            let y = (op.in_halves[3] >> 8) as u8;
            bitwise_ops.push(BitwiseOperation::halfword(
                BitwiseOperationType::Msb16,
                x,
                y,
            ));
        }

        // C1: BYTE_ALU[AND, shift, 15] | left (= μ - direction = 1 - direction)
        if left {
            bitwise_ops.push(BitwiseOperation::byte_op(
                BitwiseOperationType::ByteAluAnd,
                op.shift,
                15,
            ));
        }

        // C2: BYTE_ALU[AND, 256 - zbs*16 - shift, 15] | right (= direction)
        if right {
            let zbs_16: u16 = if aux.zbs { 16 } else { 0 };
            let complement = (256u16 - zbs_16 - op.shift as u16) as u8;
            bitwise_ops.push(BitwiseOperation::byte_op(
                BitwiseOperationType::ByteAluAnd,
                complement,
                15,
            ));
        }

        // C3: ZERO[bit_shift] | μ (= 1)
        bitwise_ops.push(BitwiseOperation::zero(aux.bit_shift as u32));

        // C4.i + C7: HWSL paired lookups | 1-zbs
        // Each HWSL lookup returns [SLL, SLLC], constraining both X[i] and Y[i]
        // from the same input in a single bus interaction.
        if !aux.zbs {
            for i in 0..4 {
                let x = (op.in_halves[i] & 0xFF) as u8;
                let y = (op.in_halves[i] >> 8) as u8;
                bitwise_ops.push(BitwiseOperation::shift_op(
                    BitwiseOperationType::Hwsl,
                    x,
                    y,
                    aux.bit_shift,
                ));
            }
            // C7: HWSL[extension, bit_shift] → [X[4], extension - X[4]]
            let extension: u16 = if aux.is_negative { 0xFFFF } else { 0 };
            let ext_x = (extension & 0xFF) as u8;
            let ext_y = (extension >> 8) as u8;
            bitwise_ops.push(BitwiseOperation::shift_op(
                BitwiseOperationType::Hwsl,
                ext_x,
                ext_y,
                aux.bit_shift,
            ));
        }

        // C11: BYTE_ALU[AND, shift, mask] | μ (= 1)
        let mask = if op.word_instr { 16 } else { 48 };
        bitwise_ops.push(BitwiseOperation::byte_op(
            BitwiseOperationType::ByteAluAnd,
            op.shift,
            mask,
        ));

        // Range checks (match the ALU-bus in2 reconstruction): ARE_BYTES[bits
        // 8-15] + IS_HALF[bits 16-31]. The high word (bits 32-63, SHIFT_HIGH) is
        // the spec's `shift[3]` Word; IS_WORD is assumed via its bus equality
        // with the CPU's well-formed arg2 high word, so it needs no check.
        bitwise_ops.push(BitwiseOperation::single_byte(
            BitwiseOperationType::AreBytes,
            ((op.shift_amount >> 8) & 0xFF) as u8,
        ));
        // ARE_BYTES[shift[0]] — spec IS_BYTE[shift[0]] (defense-in-depth,
        // redundant with the BYTE_ALU[AND, shift, mask] lookups above).
        bitwise_ops.push(BitwiseOperation::single_byte(
            BitwiseOperationType::AreBytes,
            op.shift,
        ));
        let half = ((op.shift_amount >> 16) & 0xFFFF) as u16;
        bitwise_ops.push(BitwiseOperation::halfword(
            BitwiseOperationType::IsHalf,
            (half & 0xFF) as u8,
            (half >> 8) as u8,
        ));
        // VM-3: IS_HALF[in[i]] for the four input halves, unconditional on every
        // active row — matches the four IS_HALF senders added in `bus_interactions`.
        for i in 0..4 {
            bitwise_ops.push(BitwiseOperation::halfword(
                BitwiseOperationType::IsHalf,
                (op.in_halves[i] & 0xFF) as u8,
                (op.in_halves[i] >> 8) as u8,
            ));
        }
    }

    bitwise_ops
}
