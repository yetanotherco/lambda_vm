//! SHIFT chip for bit-shift operations.
//!
//! Constrains: `out = in <</>>/>>> (shift mod (32 * (2 - word_instr)))`.
//!
//! Two-phase design:
//! 1. Intra-limb shift by `bit_shift = shift mod 16` using paired HWSL lookups (returning [SLL, SLLC]).
//! 2. Full-limb shift by `limb_shift` (unary encoding of `shift >> 4`).
//!
//! ## Columns (29 total)
//! - Input: `in[0..3]` (DWordHL), `shift` (Byte), `direction` (Bit), `signed` (Bit), `word_instr` (Bit)
//! - Output: `out[0..1]` (DWordWL)
//! - Auxiliary: `is_negative`, `bit_shift`, `zbs`, `X[0..4]`, `Y[0..3]`, `limb_shift_raw[0..2]`
//! - Virtual: `limb_shift[3] = 1 - limb_shift_raw[0] - limb_shift_raw[1] - limb_shift_raw[2]`
//! - Shift decomposition (ALU-bus shift amount): `shift_b1` (idx 26, Byte = shift[1]), `shift_h1` (idx 27, Half = shift[2]), `shift_high` (idx 28, Word = shift[3])
//! - Multiplicity: `μ`
//!
//! ## Bus Interactions (18 total)
//! - Senders: MSB16, BYTE_ALU[AND] (×3), ZERO, HWSL (×5), ARE_BYTES (×2), IS_HALFWORD (×5)
//! - Receiver: ALU (from CPU)

use stark::lookup::{BusInteraction, BusValue, LinearTerm, Multiplicity, Packing};
use stark::trace::TraceTable;

use super::types::{BusId, GoldilocksExtension, GoldilocksField, SHIFT_16, VmTable, alu_op};

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
        // so the extension contribution in `shifted_half` naturally
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
    let mut trace = TraceTable::new_main(
        crate::tables::types::zeroed_fe_vec(num_rows * cols::NUM_COLUMNS),
        cols::NUM_COLUMNS,
        1,
    );
    let table = &mut trace.main_table;

    for (row_idx, op) in operations.iter().enumerate() {
        let aux = op.compute_aux();

        // Input columns
        table.set_halves(row_idx, cols::IN_0, &op.in_halves);
        table.set_byte(row_idx, cols::SHIFT_AMOUNT, op.shift);
        // High bits of the full shift amount (for the ALU bus in2 = arg2).
        table.set_byte(
            row_idx,
            cols::SHIFT_B1,
            ((op.shift_amount >> 8) & 0xFF) as u8,
        );
        table.set_half(
            row_idx,
            cols::SHIFT_H1,
            ((op.shift_amount >> 16) & 0xFFFF) as u16,
        );
        table.set_word(row_idx, cols::SHIFT_HIGH, (op.shift_amount >> 32) as u32);
        table.set_bool(row_idx, cols::DIRECTION, op.direction);
        table.set_bool(row_idx, cols::SIGNED, op.signed);
        table.set_bool(row_idx, cols::WORD_INSTR, op.word_instr);

        // Output columns
        table.set_words(row_idx, cols::OUT_0, &aux.out);

        // Auxiliary columns
        table.set_bool(row_idx, cols::IS_NEGATIVE, aux.is_negative);
        table.set_byte(row_idx, cols::BIT_SHIFT, aux.bit_shift);
        table.set_bool(row_idx, cols::ZBS, aux.zbs);

        table.set_halves(row_idx, cols::X_0, &aux.x);
        table.set_halves(row_idx, cols::Y_0, &aux.y);
        for i in 0..3 {
            table.set_bool(row_idx, cols::LIMB_SHIFT_RAW[i], aux.limb_shift[i]);
        }
        // limb_shift[3] is virtual: not stored in the trace

        // μ = 1 for all active rows (Bit)
        table.set_bool(row_idx, cols::MU, true);
    }

    // Padding rows: set ZBS=1 per spec. All other columns remain 0.
    // μ=0 so C13 (limb_shift encoding) is inactive. left=right=0 so shifted=0,
    // making C14 (out=shifted) trivially satisfied regardless of limb_shift.
    for row_idx in operations.len()..num_rows {
        table.set_bool(row_idx, cols::ZBS, true);
    }

    trace
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
        one_minus_zbs,
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

/// Total number of SHIFT transition constraints.
pub const NUM_SHIFT_CONSTRAINTS: usize = 19;

// =========================================================================
// Single-body constraint set (ConstraintSet front-end)
// =========================================================================
//
// One body against the generic `ConstraintBuilder` serves the compiled prover
// folder, the verifier folder and IR capture. Constraint indices 0..19.

use stark::constraints::builder::{ConstraintBuilder, ConstraintSet};

/// SHIFT table constraints as a single-source [`ConstraintSet`]. No column
/// configuration is needed (the SHIFT layout is fixed via `cols`).
pub struct ShiftConstraints;

impl ShiftConstraints {
    /// `limb_shift[i]` (i = 0..2 raw, i = 3 virtual
    /// `1 - ls_raw[0] - ls_raw[1] - ls_raw[2]`).
    fn limb_shift<B: ConstraintBuilder<GoldilocksField, GoldilocksExtension>>(
        b: &B,
        i: usize,
    ) -> B::Expr {
        if i < 3 {
            b.main(0, cols::LIMB_SHIFT_RAW[i])
        } else {
            let one = b.one();
            let a = b.main(0, cols::LIMB_SHIFT_RAW[0]);
            let c = b.main(0, cols::LIMB_SHIFT_RAW[1]);
            let d = b.main(0, cols::LIMB_SHIFT_RAW[2]);
            one - a - c - d
        }
    }

    /// intra_limb_left[i]: X[0] for i=0, X[i]+Y[i-1] for i>0.
    fn intra_left<B: ConstraintBuilder<GoldilocksField, GoldilocksExtension>>(
        b: &B,
        i: usize,
    ) -> B::Expr {
        if i == 0 {
            b.main(0, cols::X[0])
        } else {
            let x = b.main(0, cols::X[i]);
            let y = b.main(0, cols::Y[i - 1]);
            x + y
        }
    }

    /// intra_limb_right[i]: Y[i]+X[i+1].
    fn intra_right<B: ConstraintBuilder<GoldilocksField, GoldilocksExtension>>(
        b: &B,
        i: usize,
    ) -> B::Expr {
        let y = b.main(0, cols::Y[i]);
        let x = b.main(0, cols::X[i + 1]);
        y + x
    }

    /// The `shifted` virtual column at index `half_idx` (0..4).
    fn shifted_half<B: ConstraintBuilder<GoldilocksField, GoldilocksExtension>>(
        b: &B,
        i: usize,
    ) -> B::Expr {
        // left = μ - direction, right = direction
        let mu = b.main(0, cols::MU);
        let dir = b.main(0, cols::DIRECTION);
        let left = mu - dir;
        let right = b.main(0, cols::DIRECTION);

        // extension = 65535 * is_negative
        let is_neg = b.main(0, cols::IS_NEGATIVE);
        let c65535 = b.const_base(65535);
        let extension = is_neg * c65535;

        // left_part = left * Σ_{j=0}^{i} limb_shift[j] * intra_limb_left[i-j]
        let mut left_part = b.zero();
        for j in 0..=i {
            left_part = left_part + Self::limb_shift(b, j) * Self::intra_left(b, i - j);
        }
        let left_part = left * left_part;

        // right_shift_part = Σ_{j=0}^{3-i} limb_shift[j] * intra_limb_right[i+j]
        let mut right_shift_part = b.zero();
        for j in 0..=(3 - i) {
            right_shift_part =
                right_shift_part + Self::limb_shift(b, j) * Self::intra_right(b, i + j);
        }

        // right_ext_part = extension * Σ_{j=4-i}^{3} limb_shift[j]
        let mut ext_sum = b.zero();
        if i < 4 {
            for j in (4 - i)..4 {
                ext_sum = ext_sum + Self::limb_shift(b, j);
            }
        }
        let right_ext_part = extension * ext_sum;

        let right_part = right * (right_shift_part + right_ext_part);

        left_part + right_part
    }
}

impl ConstraintSet<GoldilocksField, GoldilocksExtension> for ShiftConstraints {
    fn max_degree(&self) -> usize {
        crate::VM_MAX_DEGREE
    }

    fn eval<B: ConstraintBuilder<GoldilocksField, GoldilocksExtension>>(&self, b: &mut B) {
        // idx 0: DirectionImpliesMu — direction * (1 - μ)
        let dir = b.main(0, cols::DIRECTION);
        let mu = b.main(0, cols::MU);
        let one = b.one();
        b.emit_base(0, dir * (one - mu));

        // idx 1..5: ZbsOverrideX(i) — zbs * (X[i] - in[i] * (μ - direction))
        for i in 0..4 {
            let zbs = b.main(0, cols::ZBS);
            let x_i = b.main(0, cols::X[i]);
            let in_i = b.main(0, cols::IN[i]);
            let mu = b.main(0, cols::MU);
            let dir = b.main(0, cols::DIRECTION);
            let left = mu - dir;
            b.emit_base(1 + i, zbs * (x_i - in_i * left));
        }

        // idx 5: ZbsOverrideX4 — zbs * X[4]
        let zbs = b.main(0, cols::ZBS);
        let x4 = b.main(0, cols::X_4);
        b.emit_base(5, zbs * x4);

        // idx 6..10: ZbsOverrideY(i) — zbs * (Y[i] - in[i] * direction)
        for i in 0..4 {
            let zbs = b.main(0, cols::ZBS);
            let y_i = b.main(0, cols::Y[i]);
            let in_i = b.main(0, cols::IN[i]);
            let dir = b.main(0, cols::DIRECTION);
            b.emit_base(6 + i, zbs * (y_i - in_i * dir));
        }

        // idx 10..14: LimbShiftIsBit(i) — limb_shift[i] * (1 - limb_shift[i])
        for i in 0..4 {
            let ls = Self::limb_shift(b, i);
            let one = b.one();
            b.emit_base(10 + i, ls.clone() * (one - ls));
        }

        // idx 14,15: OutputMatchesShifted(i) —
        // out[i] - shifted_half[2i] - shifted_half[2i+1] * 2^16
        for i in 0..2 {
            let out_col = if i == 0 { cols::OUT_0 } else { cols::OUT_1 };
            let out = b.main(0, out_col);
            let half_lo = Self::shifted_half(b, 2 * i);
            let half_hi = Self::shifted_half(b, 2 * i + 1);
            let shift_16 = b.const_base(SHIFT_16);
            b.emit_base(14 + i, out - half_lo - half_hi * shift_16);
        }

        // idx 16..19: FlagIsBit — flag * (1 - flag) for direction, signed, word_instr
        for (off, flag_col) in [cols::DIRECTION, cols::SIGNED, cols::WORD_INSTR]
            .into_iter()
            .enumerate()
        {
            let flag = b.main(0, flag_col);
            let one = b.one();
            b.emit_base(16 + off, flag.clone() * (one - flag));
        }
    }
}

// =========================================================================
// Bitwise operation collection
// =========================================================================

use super::bitwise::{BitwiseOperation, BitwiseOperationType};

/// Collect BITWISE table lookups needed by a set of shift operations.
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
