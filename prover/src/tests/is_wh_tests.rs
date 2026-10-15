//! Tests for the IS_WH table and the IS_WHH template (spec #1003, #1015, #1017),
//! and for the KECCAK `state_ptr` that is their first caller.

use crypto::fiat_shamir::default_transcript::DefaultTranscript;
use stark::prover::{IsStarkProver, Prover};

use crate::statement::{StatementKind, absorb_statement};
use crate::tables::bitwise::{
    BitwiseHistogram, BitwiseOperation, BitwiseOperationType, cols as bw_cols,
    row_index as bw_row_index,
};
use crate::tables::is_wh::{
    self, IsWhKey, MU_MAX, collect_bitwise_from_is_wh, cols, dedup_is_wh_rows, generate_is_wh_trace,
};
use crate::tables::keccak::{
    self, KeccakConstraints, KeccakOperation, cols as keccak_cols, generate_keccak_trace,
};
use crate::tables::trace_builder::Traces;
use crate::tables::types::{BusId, FE};
use crate::test_utils::{E, asm_elf_bytes, busless_air, validate_busless};
use crate::{MaxRowsConfig, VmAirs, VmProof};

use executor::elf::Elf;
use executor::vm::execution::Executor;

const A: IsWhKey = IsWhKey {
    word: 0x0000_0001,
    half: 0x0002,
};
const B: IsWhKey = IsWhKey {
    word: 0xFFFF_FFFF,
    half: 0xFFFF,
};

fn keccak_op(state_addr: u64) -> KeccakOperation {
    KeccakOperation {
        timestamp: 7,
        state_addr,
        input: [0x0102_0304_0506_0708; 25],
        output: [0; 25],
    }
}

// =========================================================================
// Keys and dedup
// =========================================================================

#[test]
fn test_is_wh_key_of_dword_takes_the_top_48_bits() {
    let key = IsWhKey::of_dword(0x1234_5678_9ABC_DEF0);
    assert_eq!(key.word, 0x1234_5678, "word = bits 32..64");
    assert_eq!(key.half, 0x9ABC, "half = bits 16..32");
    // The low half is not part of the key: it is IS_HALF-checked by the caller.
    assert_eq!(IsWhKey::of_dword(0x1234_5678_9ABC_0000), key);
}

#[test]
fn test_is_wh_dedup_collapses_repeated_keys() {
    let rows = dedup_is_wh_rows([B, A, A, B, A]);
    // Sorted by key, so the trace does not depend on hash order.
    assert_eq!(rows, vec![(A, 3), (B, 2)]);
}

#[test]
fn test_is_wh_dedup_of_no_sends_is_empty() {
    assert!(dedup_is_wh_rows(std::iter::empty()).is_empty());
}

#[test]
fn test_is_wh_dedup_splits_mu_above_b20() {
    let total_mu = |rows: &[(IsWhKey, u64)]| rows.iter().map(|(_, mu)| mu).sum::<u64>();

    // Exactly MU_MAX fits one row.
    let rows = dedup_is_wh_rows(vec![A; MU_MAX as usize]);
    assert_eq!(rows, vec![(A, MU_MAX)]);

    // One more spills into a second row of the same key.
    let rows = dedup_is_wh_rows(vec![A; MU_MAX as usize + 1]);
    assert_eq!(rows, vec![(A, MU_MAX), (A, 1)]);

    // Two full rows and a remainder, next to an unrelated key.
    let mut keys = vec![A; 2 * MU_MAX as usize + 3];
    keys.push(B);
    let rows = dedup_is_wh_rows(keys.iter().copied());
    assert_eq!(rows, vec![(A, MU_MAX), (A, MU_MAX), (A, 3), (B, 1)]);
    assert_eq!(total_mu(&rows), keys.len() as u64, "no send lost or added");
    assert!(rows.iter().all(|(_, mu)| *mu <= MU_MAX && *mu > 0));
}

// =========================================================================
// Trace
// =========================================================================

#[test]
fn test_is_wh_trace_layout_and_padding() {
    let trace = generate_is_wh_trace(&dedup_is_wh_rows([A, B, A]));
    let t = &trace.main_table;
    assert_eq!(
        trace.num_rows(),
        4,
        "two real rows padded to the minimum of 4"
    );

    // value[0] = half, value[1..2] = word as two halves (low first).
    assert_eq!(*t.get(0, cols::VALUE_0), FE::from(A.half as u64));
    assert_eq!(*t.get(0, cols::VALUE_1), FE::from((A.word & 0xFFFF) as u64));
    assert_eq!(*t.get(0, cols::VALUE_2), FE::from((A.word >> 16) as u64));
    assert_eq!(*t.get(0, cols::MU), FE::from(2u64));
    assert_eq!(*t.get(1, cols::VALUE_0), FE::from(0xFFFFu64));
    assert_eq!(*t.get(1, cols::VALUE_1), FE::from(0xFFFFu64));
    assert_eq!(*t.get(1, cols::VALUE_2), FE::from(0xFFFFu64));
    assert_eq!(*t.get(1, cols::MU), FE::one());

    // Padding is all zero (spec pad = 0).
    for row in 2..4 {
        for col in 0..cols::NUM_COLUMNS {
            assert_eq!(*t.get(row, col), FE::zero(), "padding ({row}, {col})");
        }
    }
}

#[test]
fn test_is_wh_trace_exact_power_of_two_has_no_padding() {
    let keys: Vec<IsWhKey> = (0..8).map(|i| IsWhKey { word: i, half: 0 }).collect();
    let trace = generate_is_wh_trace(&dedup_is_wh_rows(keys));
    assert_eq!(trace.num_rows(), 8);
    for row in 0..8 {
        assert_eq!(*trace.main_table.get(row, cols::MU), FE::one());
    }
}

// =========================================================================
// BITWISE collector
// =========================================================================

fn half_op(v: u16) -> BitwiseOperation {
    BitwiseOperation::halfword(BitwiseOperationType::IsHalf, v as u8, (v >> 8) as u8)
}

fn b20_op(v: u64) -> BitwiseOperation {
    BitwiseOperation::b20(v as u8, (v >> 8) as u8, (v >> 16) as u8)
}

/// Every lookup the collector counts, one op per unit of multiplicity.
fn expected_ops(rows: &[(IsWhKey, u64)], padding: usize) -> Vec<BitwiseOperation> {
    let mut ops = Vec::new();
    for (key, mu) in rows {
        for _ in 0..*mu {
            ops.push(half_op(key.half));
            ops.push(half_op((key.word & 0xFFFF) as u16));
            ops.push(half_op((key.word >> 16) as u16));
        }
        ops.push(b20_op(*mu));
    }
    ops.extend(std::iter::repeat_n(b20_op(0), padding));
    ops
}

#[test]
fn test_is_wh_collector_counts_mu_bound_and_padding() {
    let keys = [A, B, A];
    let mut got = BitwiseHistogram::new();
    collect_bitwise_from_is_wh(&dedup_is_wh_rows(keys), &mut got);

    // Rows (A, 2), (B, 1), two padding rows each sending IS_B20[0].
    let mut want = BitwiseHistogram::new();
    want.add_ops(&expected_ops(&[(A, 2), (B, 1)], 2));
    assert!(got == want, "collector disagrees with the table's sends");
}

#[test]
fn test_is_wh_collector_counts_nothing_without_sends() {
    // No sends: `generate_optional` emits no table, so no padding either.
    let mut got = BitwiseHistogram::new();
    collect_bitwise_from_is_wh(&[], &mut got);
    assert!(got == BitwiseHistogram::new());
}

// =========================================================================
// KECCAK state_ptr as DWordWHH
// =========================================================================

/// `state_ptr[lane]` recomposed from its three limbs.
fn state_ptr_value(t: &stark::trace::TraceTable<crate::test_utils::F, E>, lane: usize) -> u64 {
    let limb = |l: usize| {
        t.main_table
            .get(0, keccak_cols::state_ptr(lane, l))
            .canonical_u64()
    };
    limb(0) | (limb(1) << 16) | (limb(2) << 32)
}

/// The ADD from `addr` to `state_ptr` has to carry across both limb boundaries
/// of a DWordWHH: into `half_mid` at 2^16 and into `word_hi` at 2^32.
#[test]
fn test_keccak_state_ptr_whh_across_limb_boundaries() {
    let air = busless_air(keccak_cols::NUM_COLUMNS, KeccakConstraints);
    for state_addr in [
        0x1000,
        0x0000_0000_FFFF_FFA0, // lanes cross 2^32 (word_hi 0 → 1)
        0x0000_FFFF_FFFF_FFA0, // lanes cross 2^48 (word_hi low half wraps)
        0x0000_0000_0000_FFF8, // lane 1 crosses 2^16 (half_mid 0 → 1)
    ] {
        let op = keccak_op(state_addr);
        let trace = generate_keccak_trace(std::slice::from_ref(&op));
        for lane in 0..25 {
            assert_eq!(
                state_ptr_value(&trace, lane),
                state_addr + 8 * lane as u64,
                "state_ptr[{lane}] for addr {state_addr:#x}"
            );
        }
        assert!(
            validate_busless(&air, &trace),
            "honest KECCAK trace must satisfy the ADD for addr {state_addr:#x}"
        );

        // The IS_WH keys are the top 48 bits of the same pointers.
        let keys: Vec<IsWhKey> = keccak::is_wh_keys(std::slice::from_ref(&op)).collect();
        assert_eq!(keys.len(), 25);
        for (lane, key) in keys.iter().enumerate() {
            assert_eq!(*key, IsWhKey::of_dword(state_addr + 8 * lane as u64));
        }
    }
}

/// A `state_ptr` whose limbs describe a different address than `addr + 8·lane`
/// is rejected by the ADD, so the IS_WH keys cannot drift from the real pointers.
#[test]
fn test_keccak_state_ptr_whh_rejects_a_shifted_word() {
    let air = busless_air(keccak_cols::NUM_COLUMNS, KeccakConstraints);
    let mut trace = generate_keccak_trace(&[keccak_op(0x1000)]);
    let col = keccak_cols::state_ptr(3, 2);
    let old = *trace.main_table.get(0, col);
    trace.main_table.set(0, col, old + FE::one());
    assert!(!validate_busless(&air, &trace));
}

// =========================================================================
// End to end: the IS_B20 bound on μ
// =========================================================================

/// Moves `delta` onto the BITWISE `IS_B20` receiver row for `value`.
fn bump_b20(traces: &mut Traces, value: u64, delta: FE) {
    let bw = &mut traces.bitwise.main_table;
    let row = bw_row_index(value as u8, (value >> 8) as u8, (value >> 16) as u8);
    let old = *bw.get(row, bw_cols::MU_IS_B20);
    bw.set(row, bw_cols::MU_IS_B20, old + delta);
}

/// Proves `test_keccak` after letting `rewrite` reshape the IS_WH rows, and
/// returns whether the verifier accepts. The prover is the malicious one: it
/// skips no check of its own, it simply builds a proof over the rewritten trace.
fn prove_keccak_with(rewrite: impl FnOnce(&mut Traces)) -> bool {
    let elf_bytes = asm_elf_bytes("test_keccak");
    let options = crate::GoldilocksCubicProofOptions::with_blowup(2).expect("blowup=2 is valid");
    let program = Elf::load(&elf_bytes).expect("ELF load");
    let logs = Executor::new(&program, vec![])
        .expect("executor construction")
        .run()
        .expect("run")
        .logs;
    let mut traces = Traces::from_elf_and_logs(
        &program,
        &logs,
        &MaxRowsConfig::default(),
        &[],
        #[cfg(feature = "disk-spill")]
        stark::storage_mode::StorageMode::Ram,
    )
    .expect("trace build");

    rewrite(&mut traces);

    let table_counts = traces.table_counts();
    let airs = VmAirs::new(
        &program,
        &options,
        false,
        &traces.page_configs,
        &table_counts,
        None,
        true,
        None,
        None,
        None,
    );
    let runtime_page_ranges = traces.runtime_page_ranges();
    let num_private_input_pages = traces
        .page_configs
        .iter()
        .filter(|c| c.is_private_input)
        .count();
    let mut transcript = DefaultTranscript::<E>::new(&[]);
    absorb_statement(
        &mut transcript,
        StatementKind::Monolithic,
        &elf_bytes,
        &traces.public_output_bytes,
        &table_counts,
        num_private_input_pages,
        &runtime_page_ranges,
        options.fri_final_poly_log_degree,
    );
    let proof = Prover::multi_prove(
        airs.air_trace_pairs(&mut traces),
        &mut transcript,
        #[cfg(feature = "disk-spill")]
        stark::storage_mode::StorageMode::Ram,
    )
    .expect("prover must build the crafted proof; a prover refusal is not a verifier rejection");
    let proof = VmProof {
        proof,
        runtime_page_ranges,
        table_counts,
        public_output: traces.public_output_bytes.clone(),
        num_private_input_pages,
    };
    crate::verify_with_options(&proof, &elf_bytes, &options, None, None)
        .expect("verify must not error")
}

/// Splits IS_WH row 0 (μ = m) into `m − moved` on row 0 and `moved` on the
/// first padding row, copying the key, and rebalances every BITWISE receiver
/// that has a row for the values involved. The IS_WH and IS_HALFWORD sums are
/// unchanged by construction (both rows carry the same key); only the IS_B20
/// bound sees the split.
fn split_first_row(traces: &mut Traces, moved: FE) {
    let t = &mut traces.is_whs[0].main_table;
    let m = t.get(0, cols::MU).canonical_u64();
    assert!(
        m >= 2,
        "test_keccak's 25 lanes must share a key for this split"
    );
    let pad = (0..t.height)
        .find(|&r| *t.get(r, cols::MU) == FE::zero())
        .expect("a padding row");
    for col in [cols::VALUE_0, cols::VALUE_1, cols::VALUE_2] {
        let v = *t.get(0, col);
        t.set(pad, col, v);
    }
    let kept = FE::from(m) - moved;
    t.set(0, cols::MU, kept);
    t.set(pad, cols::MU, moved);

    // IS_B20 ×1 per row: row 0 moved from m to `kept`, the padding row from 0
    // to `moved`. A value outside [0, 2^20) has no BITWISE row to move onto.
    // Read off the table's own sends, as the attacker would: with the bound
    // removed there is nothing to rebalance and the forgery must go through,
    // which is what keeps the negative test below falsifiable.
    let b20: u64 = BusId::IsB20.into();
    if !is_wh::bus_interactions().iter().any(|bi| bi.bus_id == b20) {
        return;
    }
    bump_b20(traces, m, -FE::one());
    bump_b20(traces, 0, -FE::one());
    for v in [kept, moved] {
        let v = v.canonical_u64();
        if v <= MU_MAX {
            bump_b20(traces, v, FE::one());
        }
    }
}

/// Control: a split into two positive multiplicities is an honest witness (the
/// same shape `dedup_is_wh_rows` produces past `MU_MAX`) and verifies. This is
/// what makes the rejection below attributable to the sign of μ alone.
#[test]
fn test_is_wh_positive_split_verifies() {
    assert!(prove_keccak_with(|t| split_first_row(t, FE::one())));
}

/// The carrier #1017 closes: a μ = −1 row with an honest twin at μ = m + 1
/// leaves the IS_WH and IS_HALFWORD sums untouched, so without the IS_B20 bound
/// the row would be free to receive range checks it never sends. The bound
/// sends `IS_B20[p − 1]`, which BITWISE has no row for.
#[test]
fn test_is_wh_negative_mu_is_rejected() {
    assert!(!prove_keccak_with(|t| split_first_row(t, -FE::one())));
}

/// The honest proof of the fixture, so the two above are not both vacuous.
#[test]
fn test_is_wh_honest_keccak_verifies() {
    assert!(prove_keccak_with(|t| {
        assert_eq!(t.is_whs.len(), 1, "test_keccak must ship an IS_WH table");
    }));
}
