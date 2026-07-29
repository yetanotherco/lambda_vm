//! Instruction column-group layouts — the single source of truth shared by
//! the compiler's group emission (Milestone A), the admission validator, and
//! the chips' preprocessed column constants (Milestone B).
//!
//! Each chip's instruction fields (addresses, opcode selectors,
//! multiplicities, pooled constants) form its *instruction column group*:
//! the leading, preprocessed columns of that chip's trace, committed once per
//! program and supplied at verify time. Value columns follow after
//! `PREP_WIDTH` and are defined by the chips.

/// `LFM_CONST` — pooled constants and immediates.
pub mod const_ {
    pub const ADDR: usize = 0;
    pub const V0: usize = 1; // .. V3 = 4
    pub const MULT: usize = 5;
    pub const PREP_WIDTH: usize = 6;
}

/// `LFM_BALU` — Goldilocks ALU.
pub mod balu {
    pub const A_ADDR: usize = 0;
    pub const B_ADDR: usize = 1;
    pub const C_ADDR: usize = 2;
    pub const OUT_ADDR: usize = 3;
    pub const SEL_ADD: usize = 4;
    pub const SEL_SUB: usize = 5;
    pub const SEL_MUL: usize = 6;
    pub const SEL_DIV: usize = 7;
    pub const SEL_MULADD: usize = 8;
    pub const MULT: usize = 9;
    pub const PREP_WIDTH: usize = 10;
    pub const NUM_SELECTORS: usize = 5;
}

/// `LFM_XALU` — Fp3 ALU (word lanes 0–2, `w³ = 2`).
pub mod xalu {
    pub const A_ADDR: usize = 0;
    pub const B_ADDR: usize = 1;
    pub const C_ADDR: usize = 2;
    pub const OUT_ADDR: usize = 3;
    pub const SEL_ADD: usize = 4;
    pub const SEL_SUB: usize = 5;
    pub const SEL_MUL: usize = 6;
    pub const SEL_DIV: usize = 7;
    pub const SEL_MULADD: usize = 8;
    pub const SEL_MULBASE: usize = 9;
    pub const MULT: usize = 10;
    pub const PREP_WIDTH: usize = 11;
    pub const NUM_SELECTORS: usize = 6;
}

/// `LFM_SELECT` — conditional cell swap.
pub mod select {
    pub const BIT_ADDR: usize = 0;
    pub const INL_ADDR: usize = 1;
    pub const INR_ADDR: usize = 2;
    pub const OUTL_ADDR: usize = 3;
    pub const OUTR_ADDR: usize = 4;
    pub const MULT_L: usize = 5;
    pub const MULT_R: usize = 6;
    pub const IS_REAL: usize = 7;
    pub const PREP_WIDTH: usize = 8;
}

/// `LFM_BITDEC` — canonical 64-bit decomposition. Per-bit pairs
/// `(BIT_ADDR_i, MULT_i)` at `2 + 2i` / `3 + 2i`.
pub mod bitdec {
    pub const IN_ADDR: usize = 0;
    pub const IS_REAL: usize = 1;
    pub const NUM_BITS: usize = 64;
    pub const fn bit_addr(i: usize) -> usize {
        2 + 2 * i
    }
    pub const fn bit_mult(i: usize) -> usize {
        3 + 2 * i
    }
    pub const PREP_WIDTH: usize = 2 + 2 * NUM_BITS; // 130
}

/// `LFM_HASH` — the hash chiplet (frozen tuple contract).
pub mod hash {
    pub const IN_ADDR0: usize = 0;
    pub const IN_ADDR1: usize = 1;
    pub const IN_ADDR2: usize = 2;
    pub const OUT_ADDR0: usize = 3;
    pub const OUT_ADDR1: usize = 4;
    pub const OUT_ADDR2: usize = 5;
    pub const MODE_C: usize = 6;
    pub const MODE_P: usize = 7;
    pub const MULT0: usize = 8;
    pub const MULT1: usize = 9;
    pub const MULT2: usize = 10;
    pub const PREP_WIDTH: usize = 11;
}

/// `LFM_KECCAK` — the keccak-f[1600] adapter: binds 13 machine words of state
/// to the production `KECCAK_RND` family's two `Keccak`-bus tokens.
///
/// A keccak lane is a `u64`, which is **not** felt-representable (values in
/// `[p, 2^64)` exist), so machine-side keccak state travels as `u32` halves,
/// one half per felt lane: 25 lanes = 50 halves = 13 words of 4 lanes, with the
/// last word's top two lanes unused (tuple constants — see `chips::keccak`).
pub mod keccak {
    /// Low 32 bits of the row's tag (matches `KECCAK_RND`'s `DWordWL` timestamp).
    pub const TAG_LO: usize = 0;
    /// High 32 bits of the row's tag.
    pub const TAG_HI: usize = 1;
    /// Machine words per keccak state: `ceil(50 / 4)`.
    pub const NUM_WORDS: usize = 13;
    /// `u32` halves per keccak state: `2 × 25`.
    pub const NUM_HALVES: usize = 50;
    /// Half slots the words provide: `4 × NUM_WORDS`. The top
    /// `WORD_SLOTS − NUM_HALVES = 2` are unused and pinned to zero on the bus.
    pub const WORD_SLOTS: usize = 4 * NUM_WORDS;
    /// Sponge rate for keccak256: 136 bytes = 17 lanes = 34 halves.
    pub const RATE_BYTES: usize = 136;
    pub const RATE_LANES: usize = 17;
    pub const BLOCK_HALVES: usize = RATE_BYTES / 4; // 34
    /// Machine words per rate block: `ceil(34 / 4)`. The top
    /// `4 * BLOCK_WORDS − BLOCK_HALVES = 2` half slots are unused.
    pub const BLOCK_WORDS: usize = BLOCK_HALVES.div_ceil(4); // 9

    pub const IN_ADDR0: usize = 2; // ..IN_ADDR12 = 14
    pub const OUT_ADDR0: usize = IN_ADDR0 + NUM_WORDS; // 15 ..27
    pub const MULT0: usize = OUT_ADDR0 + NUM_WORDS; // 28 ..40
    pub const BLOCK_ADDR0: usize = MULT0 + NUM_WORDS; // 41 ..49
    /// One-hot mode selectors. Their sum is the row's is-real flag, so a
    /// padding row (both zero) emits no bus tokens.
    pub const MODE_PERM: usize = BLOCK_ADDR0 + BLOCK_WORDS; // 50
    pub const MODE_ABSORB: usize = MODE_PERM + 1; // 51
    /// The production transcript's `sample()` finalizes, REVERSES the 32 digest
    /// bytes, absorbs the reversed bytes, and returns them — the returned
    /// challenge and the next segment's prefix are the same 32 bytes. Reversal
    /// is free here: the bus recomposes each `u32` half from byte columns
    /// anyway, so a second send with the coefficients (and half order) flipped
    /// costs two interactions and four preprocessed columns, and NO value
    /// columns. `sample()` is byte-for-byte identical pre- and post-#841, so
    /// this primitive is independent of which transcript revision is targeted.
    pub const REV_ADDR0: usize = MODE_ABSORB + 1; // 52
    pub const REV_ADDR1: usize = REV_ADDR0 + 1; // 53
    pub const REV_MULT0: usize = REV_ADDR1 + 1; // 54
    pub const REV_MULT1: usize = REV_MULT0 + 1; // 55
    pub const PREP_WIDTH: usize = REV_MULT1 + 1; // 56

    /// Machine words in a keccak256 digest: 32 bytes = 8 halves.
    pub const DIGEST_WORDS: usize = 2;

    pub const fn rev_addr(word: usize) -> usize {
        REV_ADDR0 + word
    }
    pub const fn rev_mult(word: usize) -> usize {
        REV_MULT0 + word
    }

    pub const fn in_addr(word: usize) -> usize {
        IN_ADDR0 + word
    }
    pub const fn out_addr(word: usize) -> usize {
        OUT_ADDR0 + word
    }
    /// Write-multiplicity of output word `word`.
    pub const fn mult(word: usize) -> usize {
        MULT0 + word
    }
    /// Address of rate-block word `word` (absorb rows only).
    pub const fn block_addr(word: usize) -> usize {
        BLOCK_ADDR0 + word
    }

    /// The tag a keccak row carries, as a function of its row index.
    ///
    /// SOUNDNESS: the tag is the *only* thing binding a row's request token to
    /// its reply token, so tags must be unique across real rows — with a
    /// duplicate, a prover can swap two permutations' outputs and the bus still
    /// balances (pinned empirically by `keccak_probe`'s
    /// `duplicate_tag_output_swap_accepts_demonstrating_hazard`). Making the tag
    /// the row ordinal gives uniqueness by construction, and putting it in the
    /// *preprocessed* group means the prover cannot choose it at all. The
    /// admission validator re-checks uniqueness independently — this function is
    /// the compiler's rule, not the guarantee.
    pub const fn tag_for_row(row: usize) -> u64 {
        row as u64 + 1
    }
}

/// `LFM_LANES` — word ↔ lane conversion (Pack / Unpack). Pack rows receive
/// four lane cells and send the assembled word; Unpack rows receive a word
/// and send its four lanes as base cells. The shared value columns appear in
/// both tuples, which IS the semantics — the chip has no constraints.
pub mod lanes {
    pub const WORD_ADDR: usize = 0;
    pub const LANE_ADDR0: usize = 1; // ..LANE_ADDR3 = 4
    pub const MODE_PACK: usize = 5;
    pub const MODE_UNPACK: usize = 6;
    pub const WORD_MULT: usize = 7; // write-mult of the word (Pack rows only)
    pub const LANE_MULT0: usize = 8; // ..LANE_MULT3 = 11 (Unpack rows only)
    pub const PREP_WIDTH: usize = 12;
}

/// `LFM_HINT` — arena ingestion.
pub mod hint {
    pub const OUT_ADDR: usize = 0;
    pub const MULT: usize = 1;
    pub const PREP_WIDTH: usize = 2;
}

/// `LFM_PUBLIC` — attestation output.
pub mod public {
    pub const IN_ADDR: usize = 0;
    pub const INDEX: usize = 1;
    pub const IS_REAL: usize = 2;
    pub const PREP_WIDTH: usize = 3;
}

/// `LFM_RANGE` — the fixed 2^16 lookup table (program-independent; its group
/// is materialized at commitment time, Milestone B).
pub mod range {
    pub const VALUE: usize = 0;
    pub const PREP_WIDTH: usize = 1;
    pub const NUM_ROWS: usize = 1 << 16;
}

/// Minimum padded height for any instruction column group (the in-tree
/// `.next_power_of_two().max(4)` convention).
pub const MIN_GROUP_ROWS: usize = 4;

/// Pads a real row count to its committed height.
pub fn padded_rows(real_rows: usize) -> usize {
    real_rows.next_power_of_two().max(MIN_GROUP_ROWS)
}
