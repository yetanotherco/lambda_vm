use crate::vm::{
    instruction::decoding::{ArithOp, Comparison, Instruction, LoadStoreWidth},
    instruction::sim_hash,
    instruction::sim_reduced_opening::{
        REDUCED_OPENING_QUERY_SYSCALL_NUMBER, REDUCED_OPENING_ROW_INPLACE_SYSCALL_NUMBER,
        REDUCED_OPENING_ROW_SYSCALL_NUMBER, REGISTER_RO_LAYOUT_SYSCALL_NUMBER,
        reduced_opening_query, reduced_opening_row, reduced_opening_row_inplace,
        register_ro_layout,
    },
    logs::Log,
    memory::{Memory, MemoryError},
    registers::Registers,
};
use math::field::element::FieldElement;
use math::field::extensions_goldilocks::Degree3GoldilocksExtensionField;
use math::field::goldilocks::{GOLDILOCKS_PRIME, GoldilocksElement};

const REGULAR_PC_UPDATE: u64 = 4;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum SyscallNumbers {
    // Placeholder discriminant. The actual syscall value is KECCAK_SYSCALL_NUMBER.
    KeccakPermute = 0,
    Print = 1,
    Panic = 2,
    Commit = 64,
    Halt = 93,
    // Placeholder discriminant. The actual syscall value is ECSM_SYSCALL_NUMBER.
    Ecsm = 94,
    // Placeholder discriminant. The actual syscall value is FEXT_LOAD_SYSCALL_NUMBER.
    FextLoad = 95,
    // Placeholder discriminant. The actual syscall value is FEXT_FMA_SYSCALL_NUMBER.
    FextFma = 96,
    // Placeholder discriminant. The actual syscall value is FEXT_STORE_SYSCALL_NUMBER.
    FextStore = 97,
    // Field-native hash/transcript measurement ecalls (EXPERIMENT 1). Placeholder
    // discriminants; the actual syscall values are the `SIM_*_SYSCALL_NUMBER`
    // constants below. These are TRUSTED, execute-only stubs that drive NO chip
    // (see `accelerator()` and `sim_hash_ecall()`): a stub build is never proven.
    // Discriminants start at 98 to sit above the FEXT accelerator variants
    // (95/96/97) merged in from PR #818/#831.
    SimAbsorbFelts = 98,
    SimAbsorbBytes = 99,
    SimTranscriptSample = 100,
    SimHashPair = 101,
    SimHashFelts = 102,
    // DEEP reduced-opening measurement ecalls (EXPERIMENT 2). Placeholder
    // discriminants; the actual syscall values are the REDUCED_OPENING_* consts.
    // Measurement-only stubs, not accelerators.
    ReducedOpeningRow = 103,
    ReducedOpeningQuery = 104,
    // Goldilocks inverse HINT (EXPERIMENT 5). Placeholder discriminant; the
    // actual syscall value is INV_GOLDILOCKS_HINT_SYSCALL_NUMBER. UNTRUSTED by
    // construction: the guest verifies `x * hint == 1` in-circuit and rejects a
    // wrong hint, so this is SOUND (not a trusted passthrough like the sim
    // stubs). It still drives no chip on this branch, so a build emitting it is
    // execute-only (never proven — the Print/Ecall-bus caveat).
    // Discriminants continue at 105 above the FEXT accelerator (95/96/97) and
    // the sim hash/reduced-opening variants (98-104) reconciled in the prefix.
    InvGoldilocksHint = 105,
    // Fp3 (Degree3GoldilocksExtensionField) inverse HINT (EXPERIMENT 5).
    // Placeholder discriminant; the actual syscall value is
    // INV_FP3_HINT_SYSCALL_NUMBER. Same soundness story as InvGoldilocksHint:
    // the guest verifies `ext_mul(x, hint) == 1` (one Fp3 multiply) and rejects
    // a wrong hint, so returning a wrong value can only make an honest proof
    // reject, never make a false one accept. Drives no chip; execute-only.
    InvFp3Hint = 106,
    // Merkle path-verify measurement stub (ROUND-2 increment A). Placeholder
    // discriminant; the actual syscall value is VERIFY_PATH_SYSCALL_NUMBER.
    // TRUSTED but REAL: the host walks the path and returns the true accept/
    // reject, subsuming the per-node HASH_PAIR ecalls on the verify paths. Drives
    // no chip; execute-only (never proven).
    VerifyPath = 107,
    // Transcript challenge-sampling measurement stubs (ROUND-2 increment B).
    // Placeholder discriminants; actual syscall values are SAMPLE_FELT/SAMPLE_U64
    // consts below. TRUSTED passthrough: the host runs the whole
    // sample_field_element / sample_u64 (sample() + ChaCha20 + field/rejection
    // sampling) byte-identically. Drive no chip; execute-only (never proven).
    // NOTE (grand composite): #841 supersedes these stubs (see default_transcript);
    // the variants remain for the TryFrom/accelerator match but are never emitted.
    SampleFelt = 108,
    SampleU64 = 109,
    // In-place reduced-opening ABI (ROUND-2 increment C). Placeholder
    // discriminants; actual syscall values are the REGISTER_RO_LAYOUT /
    // REDUCED_OPENING_ROW_INPLACE consts. Measurement-only, not accelerators.
    RegisterRoLayout = 110,
    ReducedOpeningRowInplace = 111,
    // Real FEXT (Fp3) accelerator ops completing the #818/#831 chip API. Placeholder
    // discriminants; the actual syscall values are FEXT_BASE_MUL/FEXT_INV_SYSCALL_NUMBER.
    // Like FextLoad/FextFma/FextStore these are REAL chips (constraint-checked AIRs),
    // not measurement stubs: FEXT_BASE_MUL is the Goldilocks×Fp3 asymmetric product
    // (3 base mults) and FEXT_INV is the witnessed Fp3 inverse (chip constrains
    // `x·inv == 1`). Discriminants continue at 112/113 above the sim variants.
    FextBaseMul = 112,
    FextInv = 113,
}

/// Syscall number for KeccakPermute (u64::MAX - 1 = 0xFFFF_FFFF_FFFF_FFFE).
///
/// Cannot be an enum discriminant because it exceeds isize::MAX.
pub const KECCAK_SYSCALL_NUMBER: u64 = u64::MAX - 1;
const KECCAK_STATE_BYTES: u64 = 25 * 8;

/// Syscall number for the ECSM (elliptic-curve scalar multiply) accelerator.
///
/// The spec uses ECALL number `-11`; interpreted as an unsigned 64-bit value that is
/// `u64::MAX - 10 = 0xFFFF_FFFF_FFFF_FFF5`, which the ECSM core table puts on the `Ecall`
/// bus as `[lo32, hi32] = [2^32 - 11, 2^32 - 1]`.
pub const ECSM_SYSCALL_NUMBER: u64 = u64::MAX - 10;

/// Syscall number for `FEXT_LOAD` (spec ECALL `-20`): load a degree-3 extension
/// field element from three registers into field-storage. Unsigned it is
/// `u64::MAX - 19`, placed on the `Ecall` bus as `[2^32 - 20, 2^32 - 1]`.
pub const FEXT_LOAD_SYSCALL_NUMBER: u64 = u64::MAX - 19;

/// Syscall number for `FEXT_FMA` (spec ECALL `-21`): compute `a*b + c` over the
/// native degree-3 Goldilocks extension. Unsigned it is `u64::MAX - 20`, placed
/// on the `Ecall` bus as `[2^32 - 21, 2^32 - 1]`.
pub const FEXT_FMA_SYSCALL_NUMBER: u64 = u64::MAX - 20;

/// Syscall number for `FEXT_STORE` (ECALL `-22`): read a degree-3 extension
/// element from field-storage and write its three coefficients to RAM (the
/// read-back companion to FEXT_LOAD). Unsigned it is `u64::MAX - 21`.
pub const FEXT_STORE_SYSCALL_NUMBER: u64 = u64::MAX - 21;

/// Syscall number for `FEXT_BASE_MUL` (ECALL `-23`): the Goldilocks×Fp3
/// asymmetric product `out = base · ext`, where `base` is a single canonical
/// Goldilocks element passed by value in x10 and `ext`/`out` are field-storage
/// handles (x11/x12). Only 3 base multiplies (`out[d] = base · ext[d]`), not a
/// lifted full extension multiply. Unsigned it is `u64::MAX - 22`, in the FEXT
/// accelerator's reserved Ecall-bus band (MAX-19..MAX-21 taken; MAX-22/-23 the
/// buffer the reduced-opening stubs were renumbered off).
pub const FEXT_BASE_MUL_SYSCALL_NUMBER: u64 = u64::MAX - 22;

/// Syscall number for `FEXT_INV` (ECALL `-24`): the witnessed Fp3 inverse. x10 =
/// input field-storage handle, x11 = output handle. The executor computes `x^-1`
/// host-side and stores it; the chip constrains `x · inv == 1` in-circuit (a
/// witnessed inverse, sound by construction — see `fext_inv` and the AIR). The
/// guest keeps the up-front zero rejection, so a legitimate call never inverts
/// zero. Unsigned it is `u64::MAX - 23`, next to FEXT_BASE_MUL.
pub const FEXT_INV_SYSCALL_NUMBER: u64 = u64::MAX - 23;

/// `2^32`. ECSM memory operands must not overflow their lower 32-bit address limb when the
/// largest per-access offset is added: the 32-byte operands reach offset +31 (last byte).
const LOW_LIMB: u64 = 1 << 32;

// Field-native hash/transcript measurement ecalls (EXPERIMENT 1). Each computes
// the correct value host-side, byte-identically to the guest software path it
// replaces (see `sim_hash.rs`), and returns in one VM cycle. They drive no chip,
// so a build using them is EXECUTE-ONLY (never proven — the same LogUp-bus caveat
// as the Print ecall). Values `u64::MAX - {2..6}` sit in the unused
// high-syscall-number band (keccak = MAX-1, ecsm = MAX-10).
pub const SIM_ABSORB_FELTS_SYSCALL_NUMBER: u64 = u64::MAX - 2;
pub const SIM_ABSORB_BYTES_SYSCALL_NUMBER: u64 = u64::MAX - 3;
pub const SIM_TRANSCRIPT_SAMPLE_SYSCALL_NUMBER: u64 = u64::MAX - 4;
pub const SIM_HASH_PAIR_SYSCALL_NUMBER: u64 = u64::MAX - 5;
pub const SIM_HASH_FELTS_SYSCALL_NUMBER: u64 = u64::MAX - 6;

/// Syscall number for the Goldilocks inverse HINT (EXPERIMENT 5). The guest
/// passes a pointer to a canonical field element in x10; the executor overwrites
/// it in place with `x^-1`. UNTRUSTED: the guest checks `x * hint == 1` and
/// rejects a wrong hint, so returning a wrong value here cannot make a false
/// proof accept — it can only make an honest one reject. Drives no chip, so a
/// build emitting it is execute-only (never proven).
pub const INV_GOLDILOCKS_HINT_SYSCALL_NUMBER: u64 = u64::MAX - 7;

/// Syscall number for the Fp3 (Degree3GoldilocksExtensionField) inverse HINT
/// (EXPERIMENT 5). The guest passes a pointer to three little-endian doublewords
/// (the raw limbs of a nonzero Fp3 element) in x10; the executor overwrites them
/// in place with the limbs of `x^-1`. UNTRUSTED: the guest checks
/// `ext_mul(x, hint) == 1` and rejects a wrong hint, so a wrong value here cannot
/// make a false proof accept — only make an honest one reject. Placed in the
/// MAX-40s band to avoid the other experiments' syscall numbers (keccak = MAX-1,
/// hash stubs = MAX-2..6, base inv = MAX-7, ecsm = MAX-10). Drives no chip, so a
/// build emitting it is execute-only (never proven).
pub const INV_FP3_HINT_SYSCALL_NUMBER: u64 = u64::MAX - 40;

/// Syscall number for the Merkle path-verify measurement stub (ROUND-2 increment
/// A). The guest passes `{leaf_hash_ptr, root_ptr, index, path_ptr, path_len,
/// out_ptr}`; the executor walks the path host-side (byte-identical to
/// `verify_merkle_path_from_leaf_hash`: `keccak256_pair` fold with index-bit
/// child ordering) and writes the REAL accept/reject byte at `out_ptr`. TRUSTED
/// passthrough that still computes the true answer, so a tampered opening still
/// rejects. Placed in the MAX-50s band (keccak = MAX-1, hash stubs = MAX-2..6,
/// base inv = MAX-7, ecsm = MAX-10, RO stubs = MAX-20/21, Fp3 inv = MAX-40).
/// Drives no chip, so a build emitting it is execute-only (never proven).
pub const VERIFY_PATH_SYSCALL_NUMBER: u64 = u64::MAX - 50;

/// Syscall numbers for the transcript challenge-sampling measurement stubs
/// (ROUND-2 increment B). `SAMPLE_FELT` (MAX-51) runs the whole Fp3
/// `sample_field_element` host-side (one `sample()` step + ChaCha20 seed +
/// rejection-sampled Fp3 element); `SAMPLE_U64` (MAX-52) runs the whole
/// `sample_u64` rejection loop. Both are TRUSTED passthroughs computing the true
/// value byte-identically (a tampered transcript still diverges and rejects).
/// Drive no chip -> execute-only (never proven).
pub const SAMPLE_FELT_SYSCALL_NUMBER: u64 = u64::MAX - 51;
pub const SAMPLE_U64_SYSCALL_NUMBER: u64 = u64::MAX - 52;

impl TryFrom<u64> for SyscallNumbers {
    type Error = ();
    fn try_from(value: u64) -> Result<Self, Self::Error> {
        match value {
            1 => Ok(SyscallNumbers::Print),
            2 => Ok(SyscallNumbers::Panic),
            64 => Ok(SyscallNumbers::Commit),
            93 => Ok(SyscallNumbers::Halt),
            v if v == KECCAK_SYSCALL_NUMBER => Ok(SyscallNumbers::KeccakPermute),
            v if v == ECSM_SYSCALL_NUMBER => Ok(SyscallNumbers::Ecsm),
            v if v == FEXT_LOAD_SYSCALL_NUMBER => Ok(SyscallNumbers::FextLoad),
            v if v == FEXT_FMA_SYSCALL_NUMBER => Ok(SyscallNumbers::FextFma),
            v if v == FEXT_STORE_SYSCALL_NUMBER => Ok(SyscallNumbers::FextStore),
            v if v == FEXT_BASE_MUL_SYSCALL_NUMBER => Ok(SyscallNumbers::FextBaseMul),
            v if v == FEXT_INV_SYSCALL_NUMBER => Ok(SyscallNumbers::FextInv),
            v if v == SIM_ABSORB_FELTS_SYSCALL_NUMBER => Ok(SyscallNumbers::SimAbsorbFelts),
            v if v == SIM_ABSORB_BYTES_SYSCALL_NUMBER => Ok(SyscallNumbers::SimAbsorbBytes),
            v if v == SIM_TRANSCRIPT_SAMPLE_SYSCALL_NUMBER => {
                Ok(SyscallNumbers::SimTranscriptSample)
            }
            v if v == SIM_HASH_PAIR_SYSCALL_NUMBER => Ok(SyscallNumbers::SimHashPair),
            v if v == SIM_HASH_FELTS_SYSCALL_NUMBER => Ok(SyscallNumbers::SimHashFelts),
            v if v == INV_GOLDILOCKS_HINT_SYSCALL_NUMBER => Ok(SyscallNumbers::InvGoldilocksHint),
            v if v == INV_FP3_HINT_SYSCALL_NUMBER => Ok(SyscallNumbers::InvFp3Hint),
            v if v == REDUCED_OPENING_ROW_SYSCALL_NUMBER => Ok(SyscallNumbers::ReducedOpeningRow),
            v if v == REDUCED_OPENING_QUERY_SYSCALL_NUMBER => {
                Ok(SyscallNumbers::ReducedOpeningQuery)
            }
            v if v == VERIFY_PATH_SYSCALL_NUMBER => Ok(SyscallNumbers::VerifyPath),
            v if v == SAMPLE_FELT_SYSCALL_NUMBER => Ok(SyscallNumbers::SampleFelt),
            v if v == SAMPLE_U64_SYSCALL_NUMBER => Ok(SyscallNumbers::SampleU64),
            v if v == REGISTER_RO_LAYOUT_SYSCALL_NUMBER => Ok(SyscallNumbers::RegisterRoLayout),
            v if v == REDUCED_OPENING_ROW_INPLACE_SYSCALL_NUMBER => {
                Ok(SyscallNumbers::ReducedOpeningRowInplace)
            }
            _ => Err(()),
        }
    }
}

/// A syscall that drives a specialized in-circuit accelerator chip.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Accelerator {
    Keccak,
    Ecsm,
    FextLoad,
    FextFma,
    FextStore,
    FextBaseMul,
    FextInv,
}

/// One of the five field-native hash/transcript measurement ecalls
/// (EXPERIMENT 1). These drive NO chip — they are trusted, execute-only stubs
/// counted separately from real accelerators so the optimistic-ceiling score
/// can be recomputed under different chip-cost assumptions.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum SimHashEcall {
    AbsorbFelts,
    AbsorbBytes,
    TranscriptSample,
    HashPair,
    HashFelts,
}

impl SyscallNumbers {
    /// The accelerator this syscall drives, if any. Exhaustive `match self`:
    /// adding a `SyscallNumbers` variant is a compile error here, so a new
    /// accelerator can't be silently missed by counters that consume this.
    pub fn accelerator(self) -> Option<Accelerator> {
        match self {
            SyscallNumbers::KeccakPermute => Some(Accelerator::Keccak),
            SyscallNumbers::Ecsm => Some(Accelerator::Ecsm),
            SyscallNumbers::FextLoad => Some(Accelerator::FextLoad),
            SyscallNumbers::FextFma => Some(Accelerator::FextFma),
            SyscallNumbers::FextStore => Some(Accelerator::FextStore),
            SyscallNumbers::FextBaseMul => Some(Accelerator::FextBaseMul),
            SyscallNumbers::FextInv => Some(Accelerator::FextInv),
            // Measurement stubs, not real accelerators: no chip table, never
            // proven. The CLI tallies them separately (see bin/cli).
            SyscallNumbers::ReducedOpeningRow
            | SyscallNumbers::ReducedOpeningQuery
            // The inverse hints drive no chip on this branch (a real chip would
            // just place them on the Ecall bus; the value is verified in-circuit).
            | SyscallNumbers::InvGoldilocksHint
            | SyscallNumbers::InvFp3Hint
            // The path-verify and transcript-sample stubs are measurement-only,
            // not chips.
            | SyscallNumbers::VerifyPath
            | SyscallNumbers::SampleFelt
            | SyscallNumbers::SampleU64
            | SyscallNumbers::RegisterRoLayout
            | SyscallNumbers::ReducedOpeningRowInplace
            | SyscallNumbers::Print
            | SyscallNumbers::Panic
            | SyscallNumbers::Commit
            | SyscallNumbers::Halt
            // The sim ecalls are measurement stubs, not chips.
            | SyscallNumbers::SimAbsorbFelts
            | SyscallNumbers::SimAbsorbBytes
            | SyscallNumbers::SimTranscriptSample
            | SyscallNumbers::SimHashPair
            | SyscallNumbers::SimHashFelts => None,
        }
    }

    /// The field-native hash/transcript measurement ecall this syscall is, if
    /// any. Exhaustive `match self` for the same "new variant = compile error"
    /// reason as [`accelerator`](Self::accelerator); the CLI tallies each of
    /// these separately under `--cycles`.
    pub fn sim_hash_ecall(self) -> Option<SimHashEcall> {
        match self {
            SyscallNumbers::SimAbsorbFelts => Some(SimHashEcall::AbsorbFelts),
            SyscallNumbers::SimAbsorbBytes => Some(SimHashEcall::AbsorbBytes),
            SyscallNumbers::SimTranscriptSample => Some(SimHashEcall::TranscriptSample),
            SyscallNumbers::SimHashPair => Some(SimHashEcall::HashPair),
            SyscallNumbers::SimHashFelts => Some(SimHashEcall::HashFelts),
            SyscallNumbers::KeccakPermute
            | SyscallNumbers::Ecsm
            // Real FEXT accelerator chips, classified by `accelerator()`, not here.
            | SyscallNumbers::FextLoad
            | SyscallNumbers::FextFma
            | SyscallNumbers::FextStore
            | SyscallNumbers::FextBaseMul
            | SyscallNumbers::FextInv
            | SyscallNumbers::Print
            | SyscallNumbers::Panic
            | SyscallNumbers::Commit
            | SyscallNumbers::Halt
            // EXPERIMENT 2 reduced-opening stubs, the EXPERIMENT 5 inverse hints,
            // and the ROUND-2 path-verify stub are counted by the CLI's own
            // classifiers, not here.
            | SyscallNumbers::ReducedOpeningRow
            | SyscallNumbers::ReducedOpeningQuery
            | SyscallNumbers::InvGoldilocksHint
            | SyscallNumbers::InvFp3Hint
            | SyscallNumbers::VerifyPath
            | SyscallNumbers::SampleFelt
            | SyscallNumbers::SampleU64
            | SyscallNumbers::RegisterRoLayout
            | SyscallNumbers::ReducedOpeningRowInplace => None,
        }
    }
}

/// Reads a 256-bit little-endian value as four doublewords at `addr + 8i`.
fn load_u256_le(memory: &Memory, addr: u64) -> Result<[u8; 32], MemoryError> {
    let mut out = [0u8; 32];
    for i in 0..4 {
        let dw = memory.load_doubleword(addr + (i as u64) * 8)?;
        out[i * 8..i * 8 + 8].copy_from_slice(&dw.to_le_bytes());
    }
    Ok(out)
}

/// Writes a 256-bit little-endian value as four doublewords at `addr + 8i`.
fn store_u256_le(memory: &mut Memory, addr: u64, bytes: &[u8; 32]) -> Result<(), MemoryError> {
    for i in 0..4 {
        let mut dw = [0u8; 8];
        dw.copy_from_slice(&bytes[i * 8..i * 8 + 8]);
        memory.store_doubleword(addr + (i as u64) * 8, u64::from_le_bytes(dw))?;
    }
    Ok(())
}

/// Checks the ECSM address-alignment assumption: `(addr mod 2^32) + max_offset < 2^32`.
fn ecsm_addr_ok(addr: u64, max_offset: u64) -> bool {
    (addr % LOW_LIMB) + max_offset < LOW_LIMB
}

/// Host-side Goldilocks inverse for the `INV_GOLDILOCKS_HINT` ecall. Returns the
/// true inverse of `x` (canonicalized), or `0` for a zero input, which has no
/// inverse. A zero (or otherwise wrong) return can never make a false proof
/// accept: the guest checks `x * hint == 1` and rejects on mismatch, and the
/// honest guest never hints a zero. Uses the same field arithmetic the guest
/// would otherwise run in-circuit, so an accepted hint is exactly `x^-1`.
fn goldilocks_inv_hint(x: u64) -> u64 {
    use math::field::goldilocks::GoldilocksField;
    use math::field::traits::IsField;
    GoldilocksField::inv(&x).unwrap_or(0)
}

/// Host-side Fp3 inverse for the `INV_FP3_HINT` ecall. `limbs` are the three raw
/// Goldilocks limbs (as the guest's `[FpE; 3]` stores them); returns the raw
/// limbs of `x^-1`, or `[0; 3]` for a zero (non-invertible) input. A zero or
/// otherwise wrong return can never make a false proof accept: the guest checks
/// `ext_mul(x, hint) == 1` and rejects on mismatch, and the honest guest never
/// hints a zero. Uses the same `Degree3GoldilocksExtensionField::inv` the guest
/// would otherwise run in-circuit, so an accepted hint is exactly `x^-1`.
fn fp3_inv_hint(limbs: [u64; 3]) -> [u64; 3] {
    use math::field::element::FieldElement;
    use math::field::extensions_goldilocks::Degree3GoldilocksExtensionField;
    use math::field::goldilocks::GoldilocksField;
    use math::field::traits::IsField;
    let a = [
        FieldElement::<GoldilocksField>::from_raw(limbs[0]),
        FieldElement::<GoldilocksField>::from_raw(limbs[1]),
        FieldElement::<GoldilocksField>::from_raw(limbs[2]),
    ];
    match Degree3GoldilocksExtensionField::inv(&a) {
        Ok(inv) => [*inv[0].value(), *inv[1].value(), *inv[2].value()],
        Err(_) => [0, 0, 0],
    }
}

/// TEST-ONLY debug gate: when `LAMBDA_VM_SIM_TAMPER_FP3_INV_HINT` is set in the
/// environment, the `INV_FP3_HINT` handler returns a deliberately WRONG inverse.
/// It exists solely to prove the guest's in-circuit check (`ext_mul(x, hint) == 1`)
/// is load-bearing: with the gate on, an honest run must reject/panic instead of
/// accepting. Read once and cached, so it never changes the cycle count (which is
/// the log length — one entry per ecall — regardless of the value returned) on the
/// honest, gate-off path.
fn fp3_inv_hint_tamper_enabled() -> bool {
    use std::sync::OnceLock;
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| std::env::var_os("LAMBDA_VM_SIM_TAMPER_FP3_INV_HINT").is_some())
}

/// Computes `a*b + c` over the native degree-3 Goldilocks extension
/// `Fp[x]/(x^3 - 2)`, returning canonical coefficients. Inputs must already be
/// canonical (`< p`). Matches `Degree3GoldilocksExtensionField::mul`, so the
/// executor and the FEXT prover chip (and its trace builder) agree bit-for-bit.
pub fn fext_fma(a: [u64; 3], b: [u64; 3], c: [u64; 3]) -> [u64; 3] {
    type Fp3 = FieldElement<Degree3GoldilocksExtensionField>;
    let to_fp3 = |x: [u64; 3]| {
        Fp3::from_raw([
            GoldilocksElement::from(x[0]),
            GoldilocksElement::from(x[1]),
            GoldilocksElement::from(x[2]),
        ])
    };
    let res = to_fp3(a) * to_fp3(b) + to_fp3(c);
    let coeffs = res.value();
    [
        coeffs[0].canonical_u64(),
        coeffs[1].canonical_u64(),
        coeffs[2].canonical_u64(),
    ]
}

/// Computes the Goldilocks×Fp3 asymmetric product `out = base · ext` for the
/// `FEXT_BASE_MUL` accelerator: `out[d] = base · ext[d]` (three base multiplies).
/// `base` must be a canonical Goldilocks element (`< p`), `ext` canonical Fp3
/// coefficients. Matches the subfield product `<GoldilocksField as
/// IsSubFieldOf<Fp3>>::mul`, so executor, chip, and guest agree bit-for-bit.
pub fn fext_base_mul(base: u64, ext: [u64; 3]) -> [u64; 3] {
    let b = GoldilocksElement::from(base);
    [
        (b * GoldilocksElement::from(ext[0])).canonical_u64(),
        (b * GoldilocksElement::from(ext[1])).canonical_u64(),
        (b * GoldilocksElement::from(ext[2])).canonical_u64(),
    ]
}

/// Computes the Fp3 multiplicative inverse `x^-1` for the `FEXT_INV` accelerator,
/// returning canonical coefficients (or `[0; 3]` for a zero input, which has no
/// inverse). The chip constrains `x · inv == 1` (with a zero flag), so an
/// accepted result is exactly `x^-1`; the honest guest rejects zero before
/// calling, so `[0; 3]` is never returned on a legitimate path. Uses the same
/// `Degree3GoldilocksExtensionField::inv` the guest would run in software.
pub fn fext_inv(x: [u64; 3]) -> [u64; 3] {
    use math::field::traits::IsField;
    let a = [
        GoldilocksElement::from(x[0]),
        GoldilocksElement::from(x[1]),
        GoldilocksElement::from(x[2]),
    ];
    match Degree3GoldilocksExtensionField::inv(&a) {
        Ok(inv) => [
            inv[0].canonical_u64(),
            inv[1].canonical_u64(),
            inv[2].canonical_u64(),
        ],
        Err(_) => [0, 0, 0],
    }
}

impl Instruction {
    /// Runs the given instruction and returns its execution log
    pub fn run(
        self,
        pc: &mut u64,
        registers: &mut Registers,
        memory: &mut Memory,
    ) -> Result<Log, ExecutionError> {
        let log = self.execute(*pc, registers, memory)?;
        *pc = log.next_pc;
        Ok(log)
    }

    /// Executes the given instruction returning the new value of pc, the register to be updated and the new value of said register
    fn execute(
        self,
        pc: u64,
        registers: &mut Registers,
        memory: &mut Memory,
    ) -> Result<Log, ExecutionError> {
        Ok(match self {
            Instruction::ArithImm { dst, src, imm, op } => {
                let op1 = registers.read(src)? as i64;
                if matches!(op, ArithOp::Sub) {
                    return Err(ExecutionError::SubImmNotSupported);
                }
                let res = op.apply(op1, imm as i64) as u64;
                registers.write(dst, res)?;
                Log {
                    current_pc: pc,
                    next_pc: pc.wrapping_add(REGULAR_PC_UPDATE),
                    src1_val: op1 as u64,
                    src2_val: 0,
                    dst_val: res,
                }
            }
            Instruction::ArithImmW { dst, src, imm, op } => {
                // W-suffix: operate on lower 32 bits, sign-extend result to 64 bits.
                // Log must store the RAW register value in src1_val (full 64 bits)
                // for the prover's MEMW register chain. The truncation to i32 is only
                // for the ALU computation.
                let raw_src = registers.read(src)?;
                let op1 = raw_src as i32;
                if matches!(op, ArithOp::Sub) {
                    return Err(ExecutionError::SubImmNotSupported);
                }
                let res32 = op.apply_word(op1, imm)?;
                let res = res32 as i64 as u64; // Sign-extend to 64 bits
                registers.write(dst, res)?;
                Log {
                    current_pc: pc,
                    next_pc: pc.wrapping_add(REGULAR_PC_UPDATE),
                    src1_val: raw_src,
                    src2_val: 0,
                    dst_val: res,
                }
            }
            Instruction::JumpAndLinkRegister { dst, base, offset } => {
                let base_value = registers.read(base)?;
                let new_pc = (((base_value as i64).wrapping_add(offset as i64)) & !1) as u64;
                registers.write(dst, pc.wrapping_add(REGULAR_PC_UPDATE))?;
                Log {
                    current_pc: pc,
                    next_pc: new_pc,
                    src1_val: base_value,
                    src2_val: 0,
                    dst_val: pc.wrapping_add(REGULAR_PC_UPDATE),
                }
            }
            Instruction::JumpAndLink { dst, offset } => {
                registers.write(dst, pc.wrapping_add(REGULAR_PC_UPDATE))?;
                Log {
                    current_pc: pc,
                    next_pc: (pc as i64).wrapping_add(offset as i64) as u64,
                    src1_val: 0,
                    src2_val: 0,
                    dst_val: pc.wrapping_add(REGULAR_PC_UPDATE),
                }
            }
            Instruction::Store {
                src,
                offset,
                base,
                width,
            } => {
                let read_value = registers.read(src)?;
                let base = registers.read(base)?;
                let addr = (base as i64).wrapping_add(offset as i64) as u64;
                match width {
                    LoadStoreWidth::Byte => {
                        let value = read_value & 0xFF;
                        memory.store_byte(addr, value as u8);
                    }
                    LoadStoreWidth::Half => {
                        let value = read_value & 0xFFFF;
                        memory.store_half(addr, value as u16)?;
                    }
                    LoadStoreWidth::Word => {
                        memory.store_word(addr, read_value as u32)?;
                    }
                    LoadStoreWidth::DoubleWord => {
                        memory.store_doubleword(addr, read_value)?;
                    }
                    LoadStoreWidth::ByteUnsigned => {
                        return Err(ExecutionError::StoreBytesUnsignedNotSupported);
                    }
                    LoadStoreWidth::HalfUnsigned => {
                        return Err(ExecutionError::StoreHalfUnsignedNotSupported);
                    }
                    LoadStoreWidth::WordUnsigned => {
                        return Err(ExecutionError::StoreWordUnsignedNotSupported);
                    }
                };
                Log {
                    current_pc: pc,
                    next_pc: pc.wrapping_add(REGULAR_PC_UPDATE),
                    src1_val: base,
                    src2_val: read_value,
                    dst_val: 0,
                }
            }
            Instruction::Load {
                dst,
                offset,
                base,
                width,
            } => {
                let base = registers.read(base)?;
                let addr = (base as i64).wrapping_add(offset as i64) as u64;
                let value = match width {
                    // RV64: LB sign-extends to 64 bits
                    LoadStoreWidth::Byte => (memory.load_byte(addr) as i8) as i64 as u64,
                    // RV64: LH sign-extends to 64 bits
                    LoadStoreWidth::Half => (memory.load_half(addr)? as i16) as i64 as u64,
                    // RV64: LW sign-extends to 64 bits
                    LoadStoreWidth::Word => (memory.load_word(addr)? as i32) as i64 as u64,
                    // RV64: LD loads 64 bits
                    LoadStoreWidth::DoubleWord => memory.load_doubleword(addr)?,
                    // RV64: LBU zero-extends to 64 bits
                    LoadStoreWidth::ByteUnsigned => memory.load_byte(addr) as u64,
                    // RV64: LHU zero-extends to 64 bits
                    LoadStoreWidth::HalfUnsigned => memory.load_half(addr)? as u64,
                    // RV64: LWU zero-extends to 64 bits
                    LoadStoreWidth::WordUnsigned => memory.load_word(addr)? as u64,
                };
                registers.write(dst, value)?;
                Log {
                    current_pc: pc,
                    next_pc: pc.wrapping_add(REGULAR_PC_UPDATE),
                    src1_val: base,
                    src2_val: 0,
                    dst_val: value,
                }
            }
            Instruction::Branch {
                src1,
                src2,
                cond,
                offset,
            } => {
                let (a, b) = (registers.read(src1)?, registers.read(src2)?);
                let new_pc = if cond.apply(a, b) {
                    (pc as i64).wrapping_add(offset as i64) as u64
                } else {
                    pc.wrapping_add(REGULAR_PC_UPDATE)
                };
                Log {
                    current_pc: pc,
                    next_pc: new_pc,
                    src1_val: a,
                    src2_val: b,
                    dst_val: 0,
                }
            }
            Instruction::LoadUpperImm { dst, imm } => {
                // RV64: LUI sign-extends the 32-bit result to 64 bits
                let value = (imm as i32) as i64 as u64;
                registers.write(dst, value)?;
                Log {
                    current_pc: pc,
                    next_pc: pc.wrapping_add(REGULAR_PC_UPDATE),
                    src1_val: 0,
                    src2_val: 0,
                    dst_val: value,
                }
            }
            Instruction::AddUpperImmToPc { dst, imm } => {
                // RV64: AUIPC adds sign-extended imm to PC
                let value = pc.wrapping_add((imm as i32) as i64 as u64);
                registers.write(dst, value)?;
                Log {
                    current_pc: pc,
                    next_pc: pc.wrapping_add(REGULAR_PC_UPDATE),
                    src1_val: 0,
                    src2_val: 0,
                    dst_val: value,
                }
            }
            Instruction::Arith {
                dst,
                src1,
                src2,
                op,
            } => {
                let a = registers.read(src1)?;
                let b = registers.read(src2)?;
                let res = op.apply(a as i64, b as i64) as u64;
                registers.write(dst, res)?;
                Log {
                    current_pc: pc,
                    next_pc: pc.wrapping_add(REGULAR_PC_UPDATE),
                    src1_val: a,
                    src2_val: b,
                    dst_val: res,
                }
            }
            Instruction::ArithW {
                dst,
                src1,
                src2,
                op,
            } => {
                // W-suffix: operate on lower 32 bits, sign-extend result to 64 bits.
                // Log must store RAW register values (full 64 bits) for the prover's
                // MEMW register chain. Truncation to i32 is only for ALU computation.
                let raw_src1 = registers.read(src1)?;
                let raw_src2 = registers.read(src2)?;
                let a = raw_src1 as i32;
                let b = raw_src2 as i32;
                let res32 = op.apply_word(a, b)?;
                let res = res32 as i64 as u64; // Sign-extend to 64 bits
                registers.write(dst, res)?;
                Log {
                    current_pc: pc,
                    next_pc: pc.wrapping_add(REGULAR_PC_UPDATE),
                    src1_val: raw_src1,
                    src2_val: raw_src2,
                    dst_val: res,
                }
            }
            Instruction::CSR {
                csr: _,
                src: _,
                dst: _,
                op: _,
            } => {
                // Todo: CSR are currently no-ops
                Log {
                    current_pc: pc,
                    next_pc: pc.wrapping_add(REGULAR_PC_UPDATE),
                    src1_val: 0,
                    src2_val: 0,
                    dst_val: 0,
                }
            }
            Instruction::EcallEbreak => {
                let syscall_number_raw = registers.read(17)?; // a7
                let syscall_number = SyscallNumbers::try_from(syscall_number_raw)
                    .map_err(|_| ExecutionError::UnknownSyscall(syscall_number_raw))?;
                let mut src2_val = 0u64;
                let mut dst_val = 0u64;
                match syscall_number {
                    SyscallNumbers::Print => {
                        // print
                        // For now this is just a mechanism to print
                        // It is not the correct implementation of ecall/ebreak
                        let pointer = registers.read(10)?;
                        let len = registers.read(11)?;
                        let bytes = memory.load_bytes(pointer, len)?;
                        let value =
                            str::from_utf8(&bytes).map_err(|_| ExecutionError::IncorrectMessage)?;
                        println!("PRINT VM: {}", value);
                    }
                    SyscallNumbers::Panic => {
                        // panic
                        let pointer = registers.read(10)?;
                        let len = registers.read(11)?;
                        let bytes = memory.load_bytes(pointer, len)?;
                        let value =
                            str::from_utf8(&bytes).map_err(|_| ExecutionError::IncorrectMessage)?;
                        return Err(ExecutionError::Panic(value.to_owned()));
                    }
                    SyscallNumbers::Commit => {
                        // commit: write(fd, buf_addr, count) per POSIX convention
                        // x10 = fd (must be 1 for stdout)
                        // x11 = buf_addr (buffer address in memory)
                        // x12 = count (number of bytes to write)
                        let fd = registers.read(10)?;
                        if fd != 1 {
                            return Err(ExecutionError::InvalidCommitFd(fd));
                        }
                        let buf_addr = registers.read(11)?;
                        let count = registers.read(12)?;
                        memory.commit_public_output(buf_addr, count)?;
                        src2_val = buf_addr;
                        dst_val = count;
                    }
                    SyscallNumbers::KeccakPermute => {
                        // keccak-f[1600] permutation on 200 bytes (25 × u64) at address in x10
                        let state_addr = registers.read(10)?;
                        if !state_addr.is_multiple_of(8) {
                            return Err(ExecutionError::UnalignedKeccakStateAddress(state_addr));
                        }
                        state_addr
                            .checked_add(KECCAK_STATE_BYTES - 1)
                            .ok_or(ExecutionError::KeccakStateAddressOverflow(state_addr))?;

                        let mut state = [0u64; 25];
                        for (i, lane) in state.iter_mut().enumerate() {
                            let lane_addr = state_addr
                                .checked_add((i as u64) * 8)
                                .ok_or(ExecutionError::KeccakStateAddressOverflow(state_addr))?;
                            *lane = memory.load_doubleword(lane_addr)?;
                        }
                        keccak_f1600(&mut state);
                        for (i, &lane) in state.iter().enumerate() {
                            let lane_addr = state_addr
                                .checked_add((i as u64) * 8)
                                .ok_or(ExecutionError::KeccakStateAddressOverflow(state_addr))?;
                            memory.store_doubleword(lane_addr, lane)?;
                        }
                        src2_val = state_addr;
                    }
                    SyscallNumbers::Ecsm => {
                        // ECSM(-11): k×G on secp256k1.
                        // x10 = addr to write xR, x11 = addr of xG, x12 = addr of k.
                        // xG, k, xR are 32-byte little-endian values; xG and xR must be
                        // canonical field elements and k must be in [1, N).
                        let addr_xr = registers.read(10)?;
                        let addr_xg = registers.read(11)?;
                        let addr_k = registers.read(12)?;
                        if !ecsm_addr_ok(addr_xg, 31)
                            || !ecsm_addr_ok(addr_xr, 31)
                            || !ecsm_addr_ok(addr_k, 31)
                        {
                            return Err(ExecutionError::EcsmAddressOverflow);
                        }
                        // xG and k must occupy disjoint 32-byte regions. The trace builder
                        // reads each operand as unaligned doubleword MEMW accesses (xG at T,
                        // k at T+1); if the regions overlap, the same address is touched at
                        // both timestamps and the MEMW consistency argument can't prove the
                        // access chain. The loaded values would still be well-defined — this
                        // guard is about trace provability, not correctness of the multiply.
                        // xR may alias either: its accesses are at a later timestamp.
                        if addr_xg.abs_diff(addr_k) < 32 {
                            return Err(ExecutionError::EcsmOperandOverlap);
                        }
                        let xg = load_u256_le(memory, addr_xg)?;
                        let k = load_u256_le(memory, addr_k)?;
                        let xr = ecsm::scalar_mul_x(&k, &xg)?;
                        store_u256_le(memory, addr_xr, &xr)?;
                        // Carry addr_xG/addr_k in the CPU log; addr_xR is recovered from x10
                        // by the ECSM register-read path in the trace builder.
                        src2_val = addr_xg;
                        dst_val = addr_k;
                    }
                    // Field-native hash/transcript measurement ecalls (EXPERIMENT 1).
                    // Each reproduces host-side, byte-identically, the guest software
                    // path it replaces (see `sim_hash`). Args are read from a0.. (x10..)
                    // in order: ABSORB_FELTS uses a0..a3, HASH_FELTS uses a0..a5. The
                    // carried log values mirror keccak/ecsm (pointers only) but are
                    // never consumed by a prover — a stub build is execute-only.
                    SyscallNumbers::SimAbsorbFelts => {
                        let state_ptr = registers.read(10)?;
                        let elems_ptr = registers.read(11)?;
                        let count = registers.read(12)?;
                        let kind = registers.read(13)?;
                        sim_hash::absorb_felts(memory, state_ptr, elems_ptr, count, kind)?;
                        src2_val = state_ptr;
                        dst_val = elems_ptr;
                    }
                    SyscallNumbers::SimAbsorbBytes => {
                        let state_ptr = registers.read(10)?;
                        let bytes_ptr = registers.read(11)?;
                        let len = registers.read(12)?;
                        sim_hash::absorb_bytes(memory, state_ptr, bytes_ptr, len)?;
                        src2_val = state_ptr;
                        dst_val = bytes_ptr;
                    }
                    SyscallNumbers::SimTranscriptSample => {
                        let state_ptr = registers.read(10)?;
                        let out_ptr = registers.read(11)?;
                        sim_hash::transcript_sample(memory, state_ptr, out_ptr)?;
                        src2_val = state_ptr;
                        dst_val = out_ptr;
                    }
                    SyscallNumbers::SimHashPair => {
                        let l_ptr = registers.read(10)?;
                        let r_ptr = registers.read(11)?;
                        let out_ptr = registers.read(12)?;
                        sim_hash::hash_pair(memory, l_ptr, r_ptr, out_ptr)?;
                        src2_val = l_ptr;
                        dst_val = out_ptr;
                    }
                    SyscallNumbers::SimHashFelts => {
                        // Two-slice leaf hash a‖b: a0=a_ptr a1=a_count a2=b_ptr
                        // a3=b_count a4=kind a5=out_ptr (b_count=0 for a single
                        // slice). Matches the verifier's `evaluations ‖
                        // evaluations_sym` leaf shape.
                        let a_ptr = registers.read(10)?;
                        let a_count = registers.read(11)?;
                        let b_ptr = registers.read(12)?;
                        let b_count = registers.read(13)?;
                        let kind = registers.read(14)?;
                        let out_ptr = registers.read(15)?;
                        sim_hash::hash_felts(
                            memory, a_ptr, a_count, b_ptr, b_count, kind, out_ptr,
                        )?;
                        src2_val = a_ptr;
                        dst_val = out_ptr;
                    }
                    SyscallNumbers::ReducedOpeningRow => {
                        // MEASUREMENT-ONLY (never proven). Level A: compute one
                        // OOD row's (base_row_sum, base_row_sum_sym) host-side.
                        // x10 = &input struct, x11 = row_idx, x12 = out_ptr.
                        let input_ptr = registers.read(10)?;
                        let row_idx = registers.read(11)?;
                        let out_ptr = registers.read(12)?;
                        reduced_opening_row(memory, input_ptr, row_idx, out_ptr)?;
                        src2_val = input_ptr;
                        dst_val = out_ptr;
                    }
                    SyscallNumbers::ReducedOpeningQuery => {
                        // MEASUREMENT-ONLY (never proven). Level B: reconstruct
                        // the whole (deep_eval, deep_eval_sym) pair host-side.
                        // x10 = &input struct, x11 = out_ptr.
                        let input_ptr = registers.read(10)?;
                        let out_ptr = registers.read(11)?;
                        reduced_opening_query(memory, input_ptr, out_ptr)?;
                        src2_val = input_ptr;
                        dst_val = out_ptr;
                    }
                    SyscallNumbers::InvGoldilocksHint => {
                        // Goldilocks inverse HINT (EXPERIMENT 5). x10 points at a
                        // canonical field element; overwrite it in place with
                        // x^-1. UNTRUSTED — the guest verifies `x * hint == 1`
                        // and rejects a wrong value, so this handler needs no
                        // chip and can't unsoundly force acceptance.
                        let ptr = registers.read(10)?;
                        let x = memory.load_doubleword(ptr)?;
                        let inv = goldilocks_inv_hint(x);
                        memory.store_doubleword(ptr, inv)?;
                        src2_val = ptr;
                        dst_val = inv;
                    }
                    SyscallNumbers::InvFp3Hint => {
                        // Fp3 inverse HINT (EXPERIMENT 5). x10 points at three
                        // consecutive little-endian doublewords (the raw limbs of
                        // an Fp3 element); overwrite them in place with the limbs
                        // of x^-1. UNTRUSTED — the guest verifies
                        // `ext_mul(x, hint) == 1` (one Fp3 multiply) and rejects a
                        // wrong value, so this handler needs no chip and can't
                        // unsoundly force acceptance.
                        let ptr = registers.read(10)?;
                        let limbs = [
                            memory.load_doubleword(ptr)?,
                            memory.load_doubleword(ptr + 8)?,
                            memory.load_doubleword(ptr + 16)?,
                        ];
                        let mut inv = fp3_inv_hint(limbs);
                        if fp3_inv_hint_tamper_enabled() {
                            // TEST-ONLY: corrupt the inverse so `ext_mul(x, hint)`
                            // differs from `1` by a nonzero field element; the
                            // guest's in-circuit check must then reject.
                            inv[0] = inv[0].wrapping_add(1);
                        }
                        memory.store_doubleword(ptr, inv[0])?;
                        memory.store_doubleword(ptr + 8, inv[1])?;
                        memory.store_doubleword(ptr + 16, inv[2])?;
                        src2_val = ptr;
                        dst_val = inv[0];
                    }
                    SyscallNumbers::VerifyPath => {
                        // MEASUREMENT-ONLY (never proven). ROUND-2 increment A:
                        // verify one Merkle inclusion path host-side and write the
                        // REAL accept/reject byte, subsuming the per-node HASH_PAIR
                        // fold. a0=leaf_hash_ptr a1=root_ptr a2=index a3=path_ptr
                        // a4=path_len a5=out_ptr.
                        let leaf_hash_ptr = registers.read(10)?;
                        let root_ptr = registers.read(11)?;
                        let index = registers.read(12)?;
                        let path_ptr = registers.read(13)?;
                        let path_len = registers.read(14)?;
                        let out_ptr = registers.read(15)?;
                        sim_hash::verify_path(
                            memory,
                            leaf_hash_ptr,
                            root_ptr,
                            index,
                            path_ptr,
                            path_len,
                            out_ptr,
                        )?;
                        src2_val = leaf_hash_ptr;
                        dst_val = out_ptr;
                    }
                    SyscallNumbers::SampleFelt => {
                        // MEASUREMENT-ONLY (never proven). ROUND-2 increment B:
                        // the whole Fp3 `sample_field_element` host-side (sample()
                        // + ChaCha20 + rejection-sampled Fp3). a0=state_ptr
                        // a1=out_ptr (writes 3 raw limbs).
                        let state_ptr = registers.read(10)?;
                        let out_ptr = registers.read(11)?;
                        sim_hash::sample_felt(memory, state_ptr, out_ptr)?;
                        src2_val = state_ptr;
                        dst_val = out_ptr;
                    }
                    SyscallNumbers::SampleU64 => {
                        // MEASUREMENT-ONLY (never proven). ROUND-2 increment B:
                        // the whole `sample_u64` rejection loop host-side.
                        // a0=state_ptr a1=upper_bound a2=out_ptr (writes 1 u64).
                        let state_ptr = registers.read(10)?;
                        let upper_bound = registers.read(11)?;
                        let out_ptr = registers.read(12)?;
                        sim_hash::sample_u64(memory, state_ptr, upper_bound, out_ptr)?;
                        src2_val = state_ptr;
                        dst_val = out_ptr;
                    }
                    SyscallNumbers::RegisterRoLayout => {
                        // MEASUREMENT-ONLY (never proven). ROUND-2 increment C:
                        // cache the proof-constant reduced-opening layout for the
                        // subsequent in-place row ecalls. a0 = &layout.
                        let layout_ptr = registers.read(10)?;
                        register_ro_layout(memory, layout_ptr)?;
                        src2_val = layout_ptr;
                        dst_val = 0;
                    }
                    SyscallNumbers::ReducedOpeningRowInplace => {
                        // MEASUREMENT-ONLY (never proven). ROUND-2 increment C:
                        // one OOD row's (base_row_sum, base_row_sum_sym) using the
                        // registered layout + the per-query eval-slice base ptrs.
                        // a0 = row_idx, a1 = evals_ptr, a2 = out_ptr.
                        let row_idx = registers.read(10)?;
                        let evals_ptr = registers.read(11)?;
                        let out_ptr = registers.read(12)?;
                        reduced_opening_row_inplace(memory, row_idx, evals_ptr, out_ptr)?;
                        src2_val = evals_ptr;
                        dst_val = out_ptr;
                    }
                    SyscallNumbers::FextLoad => {
                        // FEXT_LOAD(-20): store a degree-3 extension element into
                        // field-storage. x10 = destination field-storage address;
                        // x11/x12/x13 = the three coefficients (native u64 form,
                        // each must be a canonical Goldilocks element `< p`).
                        let addr = registers.read(10)?;
                        let mut coeffs = [0u64; 3];
                        for (i, slot) in coeffs.iter_mut().enumerate() {
                            let v = registers.read(11 + i as u32)?;
                            if v >= GOLDILOCKS_PRIME {
                                return Err(ExecutionError::FextCoefficientNotCanonical(v));
                            }
                            *slot = v;
                        }
                        memory.field_store(addr, coeffs);
                        src2_val = addr;
                    }
                    SyscallNumbers::FextFma => {
                        // FEXT_FMA(-21): output = a*b + c over Fp[x]/(x^3-2).
                        // Per spec: x10/x11/x12 = addresses of a/b/c, x13 = output
                        // address, all in field-storage.
                        let a_addr = registers.read(10)?;
                        let b_addr = registers.read(11)?;
                        let c_addr = registers.read(12)?;
                        let out_addr = registers.read(13)?;
                        // The chip uses a single timestamp for all field-storage
                        // accesses, so the four cells must be pairwise distinct:
                        // otherwise the same (domain, address) is touched twice at
                        // one timestamp and the memory argument can't prove the
                        // access chain. (This forbids in-place `out == a` and
                        // squaring `a == b`.)
                        let addrs = [out_addr, a_addr, b_addr, c_addr];
                        for i in 0..addrs.len() {
                            for j in (i + 1)..addrs.len() {
                                if addrs[i] == addrs[j] {
                                    return Err(ExecutionError::FextOperandOverlap);
                                }
                            }
                        }
                        let a = memory.field_load(a_addr);
                        let b = memory.field_load(b_addr);
                        let c = memory.field_load(c_addr);
                        memory.field_store(out_addr, fext_fma(a, b, c));
                        src2_val = a_addr;
                        dst_val = b_addr;
                    }
                    SyscallNumbers::FextStore => {
                        // FEXT_STORE(-22): read a degree-3 extension element from
                        // field-storage (a0 = source address) and write its three
                        // coefficients back to registers a1/a2/a3 (the read-back
                        // companion to FEXT_LOAD, which reads coeffs from a1/a2/a3).
                        let src_addr = registers.read(10)?;
                        let coeffs = memory.field_load(src_addr);
                        registers.write(11, coeffs[0])?;
                        registers.write(12, coeffs[1])?;
                        registers.write(13, coeffs[2])?;
                        src2_val = src_addr;
                    }
                    SyscallNumbers::FextBaseMul => {
                        // FEXT_BASE_MUL(-23): out = base · ext over Fp[x]/(x^3-2),
                        // the Goldilocks×Fp3 asymmetric product (3 base mults).
                        // x10 = base (canonical Goldilocks element, by value),
                        // x11 = ext field-storage address, x12 = output address.
                        let base = registers.read(10)?;
                        if base >= GOLDILOCKS_PRIME {
                            return Err(ExecutionError::FextCoefficientNotCanonical(base));
                        }
                        let ext_addr = registers.read(11)?;
                        let out_addr = registers.read(12)?;
                        // Single-timestamp field-storage: the read and write cells
                        // must differ, else the same (domain, address) is touched
                        // twice at one timestamp (forbids in-place `out == ext`).
                        if ext_addr == out_addr {
                            return Err(ExecutionError::FextOperandOverlap);
                        }
                        let ext = memory.field_load(ext_addr);
                        memory.field_store(out_addr, fext_base_mul(base, ext));
                        src2_val = ext_addr;
                        dst_val = out_addr;
                    }
                    SyscallNumbers::FextInv => {
                        // FEXT_INV(-24): out = x^-1 over Fp[x]/(x^3-2), the
                        // witnessed Fp3 inverse. x10 = input field-storage address,
                        // x11 = output address. The chip constrains `x · out == 1`
                        // (with a zero flag), so an accepted out is exactly x^-1.
                        let x_addr = registers.read(10)?;
                        let out_addr = registers.read(11)?;
                        if x_addr == out_addr {
                            return Err(ExecutionError::FextOperandOverlap);
                        }
                        let x = memory.field_load(x_addr);
                        memory.field_store(out_addr, fext_inv(x));
                        src2_val = x_addr;
                        dst_val = out_addr;
                    }
                    SyscallNumbers::Halt => {
                        // halt
                        return Ok(Log {
                            current_pc: pc,
                            next_pc: 0,                   // We halt by setting pc to 0
                            src1_val: syscall_number_raw, // actual a7 value for rv1
                            src2_val: 0,
                            dst_val: 0,
                        });
                    }
                }
                Log {
                    current_pc: pc,
                    next_pc: pc + REGULAR_PC_UPDATE,
                    src1_val: syscall_number_raw,
                    src2_val,
                    dst_val,
                }
            }
            Instruction::Fence => {
                // FENCE is a memory barrier - in single-threaded, in-order execution it's a no-op
                Log {
                    current_pc: pc,
                    next_pc: pc + REGULAR_PC_UPDATE,
                    src1_val: 0,
                    src2_val: 0,
                    dst_val: 0,
                }
            }
        })
    }
}

impl ArithOp {
    /// 64-bit arithmetic operations (RV64I base)
    fn apply(&self, a: i64, b: i64) -> i64 {
        match self {
            ArithOp::Add => a.wrapping_add(b),
            ArithOp::Sub => a.wrapping_sub(b),
            ArithOp::Xor => a ^ b,
            ArithOp::Or => a | b,
            ArithOp::And => a & b,
            // RV64: shift amount is 6 bits (0-63)
            ArithOp::ShiftLeftLogical => a.wrapping_shl((b as u32) & 0x3F),
            ArithOp::ShiftRightLogical => ((a as u64).wrapping_shr((b as u32) & 0x3F)) as i64,
            ArithOp::ShiftRightArith => a.wrapping_shr((b as u32) & 0x3F),
            ArithOp::SetLessThan => (a < b) as i64,
            ArithOp::SetLessThanU => ((a as u64) < (b as u64)) as i64,
            // RV64: 64×64 multiplication
            ArithOp::Mul => a.wrapping_mul(b),
            ArithOp::MulHigh => (((a as i128).wrapping_mul(b as i128)) >> 64) as i64,
            ArithOp::MulHighSignedUnsigned => {
                (((a as i128).wrapping_mul(b as u64 as i128)) >> 64) as i64
            }
            ArithOp::MulHighUnsigned => {
                (((a as u64 as u128).wrapping_mul(b as u64 as u128)) >> 64) as i64
            }
            // RV64: 64÷64 division
            ArithOp::Div => {
                if b == 0 {
                    -1i64
                } else {
                    a.wrapping_div(b)
                }
            }
            ArithOp::DivUnsigned => {
                if b == 0 {
                    u64::MAX as i64
                } else {
                    (a as u64).wrapping_div(b as u64) as i64
                }
            }
            ArithOp::Remainder => {
                if b == 0 {
                    a
                } else {
                    a.wrapping_rem(b)
                }
            }
            ArithOp::RemainderUnsigned => {
                if b == 0 {
                    a
                } else {
                    (a as u64).wrapping_rem(b as u64) as i64
                }
            }
        }
    }

    /// 32-bit arithmetic operations with sign extension (RV64 W-suffix)
    fn apply_word(&self, a: i32, b: i32) -> Result<i32, ExecutionError> {
        Ok(match self {
            ArithOp::Add => a.wrapping_add(b),
            ArithOp::Sub => a.wrapping_sub(b),
            // W-suffix shifts use 5-bit shift amount
            ArithOp::ShiftLeftLogical => a.wrapping_shl((b as u32) & 0x1F),
            ArithOp::ShiftRightLogical => ((a as u32).wrapping_shr((b as u32) & 0x1F)) as i32,
            ArithOp::ShiftRightArith => a.wrapping_shr((b as u32) & 0x1F),
            // MULW: 32×32→32 (low bits), sign-extend
            ArithOp::Mul => a.wrapping_mul(b),
            // DIVW: 32÷32
            ArithOp::Div => {
                if b == 0 {
                    -1i32
                } else {
                    a.wrapping_div(b)
                }
            }
            ArithOp::DivUnsigned => {
                if b == 0 {
                    u32::MAX as i32
                } else {
                    (a as u32).wrapping_div(b as u32) as i32
                }
            }
            ArithOp::Remainder => {
                if b == 0 {
                    a
                } else {
                    a.wrapping_rem(b)
                }
            }
            ArithOp::RemainderUnsigned => {
                if b == 0 {
                    a
                } else {
                    (a as u32).wrapping_rem(b as u32) as i32
                }
            }
            // These operations are not valid for W-suffix instructions
            _ => return Err(ExecutionError::InvalidWSuffixOperation(*self)),
        })
    }
}

impl Comparison {
    fn apply(&self, a: u64, b: u64) -> bool {
        match self {
            Comparison::Equal => a == b,
            Comparison::NotEqual => a != b,
            Comparison::LessThan => (a as i64) < (b as i64),
            Comparison::GreaterOrEqual => (a as i64) >= (b as i64),
            Comparison::LessThanUnsigned => a < b,
            Comparison::GreaterOrEqualUnsigned => a >= b,
        }
    }
}

#[derive(thiserror::Error, Debug)]
pub enum ExecutionError {
    #[error("Sub immediate instruction is not supported")]
    SubImmNotSupported,
    #[error("Store bytes unsigned instruction is not supported")]
    StoreBytesUnsignedNotSupported,
    #[error("Store half unsigned instruction is not supported")]
    StoreHalfUnsignedNotSupported,
    #[error("Store word unsigned instruction is not supported")]
    StoreWordUnsignedNotSupported,
    #[error("Memory error: {0}")]
    MemoryError(#[from] crate::vm::memory::MemoryError),
    #[error("Register error: {0}")]
    RegisterError(#[from] crate::vm::registers::RegisterError),
    #[error("Unknown syscall number: {0}")]
    UnknownSyscall(u64),
    #[error("Panic called with message: {0}")]
    Panic(String),
    #[error("Incorrect message encoding")]
    IncorrectMessage,
    #[error("Invalid W-suffix operation: {0:?}")]
    InvalidWSuffixOperation(ArithOp),
    #[error("Invalid commit fd: expected 1 (stdout), got {0}")]
    InvalidCommitFd(u64),
    #[error("Unaligned Keccak state address: {0:#018x}")]
    UnalignedKeccakStateAddress(u64),
    #[error("Keccak state address range overflows: {0:#018x}")]
    KeccakStateAddressOverflow(u64),
    #[error("ECSM address range overflows the lower 32-bit limb")]
    EcsmAddressOverflow,
    #[error("ECSM xG and k operand ranges overlap")]
    EcsmOperandOverlap,
    #[error("ECSM scalar multiplication error: {0}")]
    Ecsm(#[from] ecsm::EcsmError),
    #[error("sim-hash ecall: unaligned 8-byte address: {0:#018x}")]
    SimHashUnalignedAddress(u64),
    #[error("sim-hash ecall: sponge offset out of range [0, 136): {0}")]
    SimHashInvalidState(u64),
    #[error("sim-hash ecall: element limb count (kind) out of range [1, 3]: {0}")]
    SimHashInvalidKind(u64),
    #[error("sim-hash ecall: address range overflows")]
    SimHashAddressOverflow,
    #[error("sample_u64 stub: upper_bound must be greater than 0")]
    SimSampleU64ZeroBound,
    #[error("reduced-opening stub: inconsistent input dimensions")]
    SimReducedOpeningInvalidDims,
    #[error("reduced-opening in-place stub: row ecall before REGISTER_RO_LAYOUT")]
    SimReducedOpeningNoLayout,
    #[error("reduced-opening stub: non-invertible denominator")]
    SimReducedOpeningInverse,
    #[error("FEXT_LOAD coefficient is not a canonical field element: {0:#018x}")]
    FextCoefficientNotCanonical(u64),
    #[error("FEXT_FMA operand/output addresses must be pairwise distinct")]
    FextOperandOverlap,
}

// =============================================================================
// Keccak-f[1600] permutation
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
pub const KECCAK_RHO: [[u32; 5]; 5] = [
    [0, 36, 3, 41, 18],
    [1, 44, 10, 45, 2],
    [62, 6, 43, 15, 61],
    [28, 55, 25, 21, 56],
    [27, 20, 39, 8, 14],
];

/// Apply the Keccak-f[1600] permutation (24 rounds) to a 25-word state.
///
/// The state is indexed as `state[x + 5*y]` where `x, y ∈ {0..4}`.
pub fn keccak_f1600(state: &mut [u64; 25]) {
    for &rc in &KECCAK_RC {
        // θ (theta)
        let mut c = [0u64; 5];
        for x in 0..5 {
            c[x] = state[x] ^ state[x + 5] ^ state[x + 10] ^ state[x + 15] ^ state[x + 20];
        }
        let mut d = [0u64; 5];
        for x in 0..5 {
            d[x] = c[(x + 4) % 5] ^ c[(x + 1) % 5].rotate_left(1);
        }
        for x in 0..5 {
            for y in 0..5 {
                state[x + 5 * y] ^= d[x];
            }
        }

        // ρ (rho) and π (pi)
        let mut b = [0u64; 25];
        for x in 0..5 {
            for y in 0..5 {
                b[y + 5 * ((2 * x + 3 * y) % 5)] = state[x + 5 * y].rotate_left(KECCAK_RHO[x][y]);
            }
        }

        // χ (chi)
        for x in 0..5 {
            for y in 0..5 {
                state[x + 5 * y] =
                    b[x + 5 * y] ^ (!b[(x + 1) % 5 + 5 * y] & b[(x + 2) % 5 + 5 * y]);
            }
        }

        // ι (iota)
        state[0] ^= rc;
    }
}

#[cfg(test)]
mod inv_hint_tests {
    use super::*;
    use math::field::element::FieldElement;
    use math::field::extensions_goldilocks::Degree3GoldilocksExtensionField as Fp3;
    use math::field::goldilocks::GoldilocksField;
    use math::field::traits::IsField;

    type FpE = FieldElement<GoldilocksField>;

    fn fp3(limbs: [u64; 3]) -> [FpE; 3] {
        [
            FpE::from_raw(limbs[0]),
            FpE::from_raw(limbs[1]),
            FpE::from_raw(limbs[2]),
        ]
    }

    /// Drives the `INV_FP3_HINT` ecall through the real `EcallEbreak` dispatch and
    /// returns the three inverse limbs the handler wrote back to guest memory.
    fn run_fp3_inv_hint(limbs: [u64; 3]) -> [u64; 3] {
        let ptr: u64 = 0x4000;
        let mut memory = Memory::default();
        memory.store_doubleword(ptr, limbs[0]).unwrap();
        memory.store_doubleword(ptr + 8, limbs[1]).unwrap();
        memory.store_doubleword(ptr + 16, limbs[2]).unwrap();

        let mut registers = Registers::default();
        registers.write(10, ptr).unwrap(); // a0 = pointer to the element
        registers.write(17, INV_FP3_HINT_SYSCALL_NUMBER).unwrap(); // a7 = syscall

        let mut pc = 0u64;
        Instruction::EcallEbreak
            .run(&mut pc, &mut registers, &mut memory)
            .unwrap();

        [
            memory.load_doubleword(ptr).unwrap(),
            memory.load_doubleword(ptr + 8).unwrap(),
            memory.load_doubleword(ptr + 16).unwrap(),
        ]
    }

    /// The handler returns the true inverse: `x * hint == 1`, so the honest guest's
    /// in-circuit check passes and it accepts.
    #[test]
    fn fp3_inv_hint_dispatch_returns_true_inverse() {
        for limbs in [
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
            [7, 11, 13],
            [0x1234_5678_9abc_def0, 0xdead_beef, 0x00ff_00ff_00ff_00ff],
            [GoldilocksField::from_u64(u64::MAX), 42, 0],
        ] {
            let hint = run_fp3_inv_hint(limbs);
            let prod = <Fp3 as IsField>::mul(&fp3(limbs), &fp3(hint));
            assert!(
                <Fp3 as IsField>::eq(&prod, &<Fp3 as IsField>::one()),
                "handler must return x^-1 for {limbs:?}"
            );
        }
    }

    /// The guest's check is LOAD-BEARING: any hint other than the true inverse
    /// fails `ext_mul(x, hint) == 1`. We perturb each returned inverse the exact
    /// way the `LAMBDA_VM_SIM_TAMPER_FP3_INV_HINT` gate does (`limb0 += 1`) and
    /// confirm the check the guest runs would reject it. A guest running this
    /// check therefore panics on a tampered hint instead of accepting.
    #[test]
    fn fp3_inv_hint_check_rejects_tampered_inverse() {
        for limbs in [[7, 11, 13], [0x1234_5678_9abc_def0, 0xdead_beef, 3]] {
            let mut hint = run_fp3_inv_hint(limbs);
            // Sanity: the untampered hint passes.
            let ok = <Fp3 as IsField>::mul(&fp3(limbs), &fp3(hint));
            assert!(<Fp3 as IsField>::eq(&ok, &<Fp3 as IsField>::one()));

            hint[0] = hint[0].wrapping_add(1); // the debug gate's corruption
            let bad = <Fp3 as IsField>::mul(&fp3(limbs), &fp3(hint));
            assert!(
                !<Fp3 as IsField>::eq(&bad, &<Fp3 as IsField>::one()),
                "tampered hint must fail the guest's ext_mul == 1 check for {limbs:?}"
            );
        }
    }
}
