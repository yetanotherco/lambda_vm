//! Tests for the IS_WH table and the IS_WHH template (spec #1003, #1015, #1017),
//! and for the KECCAK `state_ptr` that is their first caller.

use stark::prover::IsStarkProver;

use crate::statement::{StatementKind, absorb_statement};
use crate::tables::bitwise::{
    BitwiseHistogram, BitwiseOperation, BitwiseOperationType, NUM_ROWS as BW_NUM_ROWS,
    cols as bw_cols, row_index as bw_row_index,
};
use crate::tables::is_wh::{
    self, IS_WH_MU_MAX, IsWhKey, collect_bitwise_from_is_wh, cols, dedup_is_wh_rows,
    generate_is_wh_trace,
};
use crate::tables::keccak::{
    self, KeccakConstraints, KeccakOperation, cols as keccak_cols, generate_keccak_trace,
};
use crate::tables::trace_builder::Traces;
use crate::tables::types::{BusId, FE};
use crate::test_utils::{E, asm_elf_bytes, busless_air, validate_busless};
use crate::tests::prove_elfs_tests::{BusOutcome, weigh_the_bus_with};
use crate::tests::skip_empty_tables_tests::assert_only_the_bus_rejected;
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

    // Exactly IS_WH_MU_MAX fits one row.
    let rows = dedup_is_wh_rows(vec![A; IS_WH_MU_MAX as usize]);
    assert_eq!(rows, vec![(A, IS_WH_MU_MAX)]);

    // One more spills into a second row of the same key.
    let rows = dedup_is_wh_rows(vec![A; IS_WH_MU_MAX as usize + 1]);
    assert_eq!(rows, vec![(A, IS_WH_MU_MAX), (A, 1)]);

    // Two full rows and a remainder, next to an unrelated key.
    let mut keys = vec![A; 2 * IS_WH_MU_MAX as usize + 3];
    keys.push(B);
    let rows = dedup_is_wh_rows(keys.iter().copied());
    assert_eq!(
        rows,
        vec![(A, IS_WH_MU_MAX), (A, IS_WH_MU_MAX), (A, 3), (B, 1)]
    );
    assert_eq!(total_mu(&rows), keys.len() as u64, "no send lost or added");
    assert!(rows.iter().all(|(_, mu)| *mu <= IS_WH_MU_MAX && *mu > 0));
}

/// The split point is the top of the `IS_B20` range: BITWISE row `i` receives
/// `IS_B20[i]`, so the last row takes `IS_WH_MU_MAX` and there is none past it.
/// A larger split point would build rows whose bound has no receiver.
#[test]
fn test_is_wh_mu_max_is_the_top_of_b20() {
    let (x, y, z) = (
        IS_WH_MU_MAX as u8,
        (IS_WH_MU_MAX >> 8) as u8,
        (IS_WH_MU_MAX >> 16) as u8,
    );
    assert_eq!(bw_row_index(x, y, z), BW_NUM_ROWS - 1);
    assert_eq!(IS_WH_MU_MAX + 1, BW_NUM_ROWS as u64);
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

/// Runs the asm program `name` and builds its traces with the full
/// preprocessed BITWISE table, the one a range-check forgery has to get past.
fn full_traces(name: &str) -> (Vec<u8>, Elf, Traces) {
    let elf_bytes = asm_elf_bytes(name);
    let program = Elf::load(&elf_bytes).expect("ELF load");
    let logs = Executor::new(&program, vec![])
        .expect("executor construction")
        .run()
        .expect("run")
        .logs;
    let traces = Traces::from_elf_and_logs(&program, &logs, &MaxRowsConfig::default(), &[])
        .expect("trace build");
    (elf_bytes, program, traces)
}

/// Proves `test_keccak` after letting `rewrite` reshape the IS_WH rows, and
/// returns whether the verifier accepts. The prover is the malicious one: it
/// skips no check of its own, it simply builds a proof over the rewritten trace.
fn prove_keccak_with(rewrite: impl FnOnce(&mut Traces)) -> bool {
    prove_asm_with("test_keccak", rewrite)
}

/// [`prove_keccak_with`] for any asm program.
fn prove_asm_with(name: &str, rewrite: impl FnOnce(&mut Traces)) -> bool {
    let options = crate::GoldilocksCubicProofOptions::with_blowup(2).expect("blowup=2 is valid");
    let (elf_bytes, program, mut traces) = full_traces(name);

    rewrite(&mut traces);

    let table_counts = traces.table_counts();
    let airs = VmAirs::new(
        &program,
        &options,
        false,
        &traces.page_configs,
        &table_counts,
        &crate::cpu_heights_from_traces(&traces),
        None,
        true,
        None,
        // The prover commits to its own REGISTER trace. These rewrites leave the
        // final register file halted, so this is the root the verifier recomputes.
        crate::test_utils::trace_register_commitment(
            &options,
            &crate::tables::register::register_init_from_entry_point(program.entry_point),
            &traces,
        )
        .preprocessed(),
    );
    let runtime_page_ranges = traces.runtime_page_ranges();
    let num_private_input_pages = traces
        .page_configs
        .iter()
        .filter(|c| c.is_private_input)
        .count();
    let mut transcript = crate::hash_pin::block_transcript(&[]);
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
    let proof = crate::hash_pin::BlockProver::multi_prove(
        airs.air_trace_pairs(&mut traces),
        &mut transcript,
        stark::residency_mode::ResidencyMode::Retain,
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

/// The same forgery on `test_keccak`, weighed instead of only verified: tells a
/// rejection by the bus balance apart from one anywhere else in the verifier
/// (see `assert_only_the_bus_rejected`).
fn weigh_keccak_with(rewrite: impl FnOnce(&mut Traces)) -> BusOutcome {
    let (_, program, mut traces) = full_traces("test_keccak");
    rewrite(&mut traces);
    weigh_the_bus_with(&program, &mut traces, true, false)
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
        if v <= IS_WH_MU_MAX {
            bump_b20(traces, v, FE::one());
        }
    }
}

/// Control: a split into two positive multiplicities is an honest witness (the
/// same shape `dedup_is_wh_rows` produces past `IS_WH_MU_MAX`) and verifies.
/// This is what makes the rejection below attributable to the sign of μ alone.
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

// =========================================================================
// End to end: the IS_HALFWORD checks on the key
// =========================================================================
//
// A key limb is only bound by IS_WH's own IS_HALFWORD sends: the `IS_WH` bus
// carries the Word packed (`value[1] + 2^16·value[2]`) and KECCAK's memory sends
// carry `half_lo + 2^16·half_mid`, so a non-canonical split of either sum keeps
// both buses and the ADD balanced. The forgeries below rebalance BITWISE from
// the IS_HALFWORD sends IS_WH declares, as the attacker would: with a send
// removed (and its count dropped from the collector) there is nothing to
// rebalance for that limb and the forgery must go through, which is what keeps
// the rejections falsifiable.

/// IS_WH columns that IS_WH sends an `IS_HALFWORD` check on.
fn is_wh_half_checked_columns() -> Vec<usize> {
    let half: u64 = BusId::IsHalfword.into();
    is_wh::bus_interactions()
        .iter()
        .filter(|bi| bi.bus_id == half && bi.is_sender)
        .filter_map(|bi| match bi.values.as_slice() {
            [stark::lookup::BusValue::Packed { start_column, .. }] => Some(*start_column),
            _ => None,
        })
        .collect()
}

/// Moves `delta` onto the BITWISE `IS_HALFWORD` receiver row for `value`. A
/// value outside `[0, 2^16)` has no row to move onto, so it is left unbalanced.
fn bump_half(traces: &mut Traces, value: FE, delta: FE) {
    let v = value.canonical_u64();
    if v >= 1 << 16 {
        return;
    }
    let bw = &mut traces.bitwise.main_table;
    let row = bw_row_index(v as u8, (v >> 8) as u8, 0);
    let old = *bw.get(row, bw_cols::MU_IS_HALF);
    bw.set(row, bw_cols::MU_IS_HALF, old + delta);
}

/// `lo + 2^16·hi` split again with `lo` replaced by `new_lo`: `hi` absorbs the
/// difference, so the pair packs to the same field element. Unless
/// `new_lo == lo`, the new `hi` is far outside `[0, 2^16)`.
fn resplit_keeping_lo(lo: FE, hi: FE, new_lo: FE) -> (FE, FE) {
    let inv_2_16 = FE::from(1u64 << 16).inv().expect("2^16 is invertible");
    (new_lo, hi + (lo - new_lo) * inv_2_16)
}

/// Rewrites the Word of IS_WH row 0 into a non-canonical pair of halves with
/// the same packing, so it still receives the honest `IS_WH` sends. The limb at
/// `out_of_range` (`VALUE_1` or `VALUE_2`) leaves `[0, 2^16)`; the other limb
/// moves to a different in-range value. `None` rewrites the row to itself
/// through the same steps.
fn resplit_word(traces: &mut Traces, out_of_range: Option<usize>) {
    let t = &mut traces.is_whs[0].main_table;
    let m = *t.get(0, cols::MU);
    let (v1, v2) = (*t.get(0, cols::VALUE_1), *t.get(0, cols::VALUE_2));
    let flip = |v: FE| FE::from(v.canonical_u64() ^ 1);
    let (n1, n2) = match out_of_range {
        None => (v1, v2),
        Some(c) if c == cols::VALUE_2 => resplit_keeping_lo(v1, v2, flip(v1)),
        Some(_) => {
            // `v1 + 2^16·v2` with `v2` replaced: `v1` absorbs `2^16·(v2 − v2')`.
            let n2 = flip(v2);
            (v1 + (v2 - n2) * FE::from(1u64 << 16), n2)
        }
    };
    assert_eq!(
        n1 + n2 * FE::from(1u64 << 16),
        v1 + v2 * FE::from(1u64 << 16)
    );
    t.set(0, cols::VALUE_1, n1);
    t.set(0, cols::VALUE_2, n2);

    for col in is_wh_half_checked_columns() {
        let (old, new) = match col {
            c if c == cols::VALUE_1 => (v1, n1),
            c if c == cols::VALUE_2 => (v2, n2),
            _ => continue,
        };
        bump_half(traces, old, -m);
        bump_half(traces, new, m);
    }
}

/// Control: the rewrite path of the test below with the Word left canonical
/// verifies, so its BITWISE rebalancing is what an honest witness needs.
#[test]
fn test_is_wh_canonical_word_rewrite_verifies() {
    assert!(prove_keccak_with(|t| resplit_word(t, None)));
}

/// An IS_WH row whose Word is split into halves that pack to the honest Word
/// but are not both halfwords. It receives the honest `IS_WH` sends, so only its
/// own `IS_HALFWORD` check on the bad limb can reject it — once per Word limb.
#[test]
fn test_is_wh_noncanonical_word_is_rejected() {
    for limb in [cols::VALUE_1, cols::VALUE_2] {
        let what = format!("an IS_WH row with column {limb} out of range");
        assert!(
            !prove_keccak_with(|t| resplit_word(t, Some(limb))),
            "{what} must not verify"
        );
        assert_only_the_bus_rejected(&weigh_keccak_with(|t| resplit_word(t, Some(limb))), &what);
    }
}

/// Moves lane 0 of the real KECCAK row onto its own IS_WH row: IS_WH row 0
/// gives up one send (μ = m − 1) and the first padding row takes it (μ = 1).
/// With `noncanonical`, lane 0's `state_ptr` is also resplit as
/// `(half_lo ^ 1, half_mid')` with the same `half_lo + 2^16·half_mid`, so the
/// ADD and the memory address are untouched while `half_mid'` leaves
/// `[0, 2^16)`; the new IS_WH row carries that key.
fn move_lane_0_to_own_row(traces: &mut Traces, noncanonical: bool) {
    let k = &mut traces.keccaks[0].main_table;
    assert_eq!(
        *k.get(0, keccak_cols::MU),
        FE::one(),
        "KECCAK row 0 is real"
    );
    let (lo_col, mid_col) = (keccak_cols::state_ptr(0, 0), keccak_cols::state_ptr(0, 1));
    let (lo, mid) = (*k.get(0, lo_col), *k.get(0, mid_col));
    let (new_lo, new_mid) = if noncanonical {
        resplit_keeping_lo(lo, mid, FE::from(lo.canonical_u64() ^ 1))
    } else {
        (lo, mid)
    };
    k.set(0, lo_col, new_lo);
    k.set(0, mid_col, new_mid);
    // KECCAK's own IS_HALFWORD[half_lo] (the IS_WHH template): both in range.
    bump_half(traces, lo, -FE::one());
    bump_half(traces, new_lo, FE::one());

    let t = &mut traces.is_whs[0].main_table;
    assert_eq!(
        *t.get(0, cols::VALUE_0),
        mid,
        "lane 0 sends IS_WH row 0's key"
    );
    let m = t.get(0, cols::MU).canonical_u64();
    assert!(m >= 2, "test_keccak's 25 lanes must share a key");
    let pad = (0..t.height)
        .find(|&r| *t.get(r, cols::MU) == FE::zero())
        .expect("a padding row");
    let row_0: Vec<FE> = (0..cols::MU).map(|c| *t.get(0, c)).collect();
    let mut new_row = row_0.clone();
    new_row[cols::VALUE_0] = new_mid;
    for (c, v) in new_row.iter().enumerate() {
        t.set(pad, c, *v);
    }
    t.set(0, cols::MU, FE::from(m - 1));
    t.set(pad, cols::MU, FE::one());

    let b20: u64 = BusId::IsB20.into();
    if is_wh::bus_interactions().iter().any(|bi| bi.bus_id == b20) {
        bump_b20(traces, m, -FE::one());
        bump_b20(traces, m - 1, FE::one());
        bump_b20(traces, 0, -FE::one());
        bump_b20(traces, 1, FE::one());
    }
    for col in is_wh_half_checked_columns() {
        bump_half(traces, row_0[col], -FE::one());
        bump_half(traces, new_row[col], FE::one());
    }
}

/// Control: giving one lane its own row with the honest key is an honest
/// witness and verifies, so the rejection below is the resplit alone.
#[test]
fn test_keccak_lane_on_its_own_is_wh_row_verifies() {
    assert!(prove_keccak_with(|t| move_lane_0_to_own_row(t, false)));
}

/// A `state_ptr` resplit as `(half_lo ^ 1, half_mid')` that keeps
/// `half_lo + 2^16·half_mid`: the ADD, the memory sends and KECCAK's
/// `IS_HALFWORD[half_lo]` all still hold, and IS_WH has a row for the new key.
/// Only that row's `IS_HALFWORD[value[0]]` sees `half_mid'` out of range.
#[test]
fn test_keccak_noncanonical_state_ptr_is_rejected() {
    let what = "a KECCAK state_ptr with half_mid out of range";
    assert!(
        !prove_keccak_with(|t| move_lane_0_to_own_row(t, true)),
        "{what} must not verify"
    );
    assert_only_the_bus_rejected(
        &weigh_keccak_with(|t| move_lane_0_to_own_row(t, true)),
        what,
    );
}

// =========================================================================
// End to end: more than one key, continuations and recursion
// =========================================================================

/// `test_keccak_64k` puts the state right across a 64 KiB boundary, so its
/// lanes carry two keys: IS_WH ships two real rows, 12 lanes below the
/// boundary and 13 above, and the proof verifies.
#[test]
fn test_keccak_state_across_64k_boundary_verifies() {
    assert!(prove_asm_with("test_keccak_64k", |t| {
        assert_eq!(t.is_whs.len(), 1);
        let is_wh = &t.is_whs[0].main_table;
        let rows: Vec<(u64, u64)> = (0..is_wh.height)
            .map(|r| {
                (
                    is_wh.get(r, cols::VALUE_0).canonical_u64(),
                    is_wh.get(r, cols::MU).canonical_u64(),
                )
            })
            .filter(|&(_, mu)| mu > 0)
            .collect();
        assert_eq!(rows.len(), 2, "two keys: {rows:?}");
        assert_eq!(rows[1].0, (rows[0].0 + 1) & 0xFFFF, "adjacent halves");
        assert_eq!((rows[0].1, rows[1].1), (12, 13));
    }));
}

/// Continuations ship IS_WH per epoch: the epoch that runs the keccak ecall
/// has an IS_WH table and the others do not, and the bundle verifies.
#[test]
fn test_keccak_continuation_carries_is_wh_in_its_epoch() {
    let elf_bytes = asm_elf_bytes("test_keccak_64k");
    let opts = crate::recursion::MIN_PROOF_OPTIONS;
    let bundle = crate::continuation::prove_continuation(&elf_bytes, &[], 4, &opts)
        .expect("continuation prove");
    let is_wh: Vec<usize> = bundle
        .epoch_table_counts()
        .iter()
        .map(|c| c.is_wh)
        .collect();
    assert!(is_wh.len() > 1, "16-cycle epochs must split the program");
    assert_eq!(
        is_wh.iter().filter(|&&n| n == 1).count(),
        1,
        "exactly the keccak epoch ships IS_WH: {is_wh:?}"
    );
    assert!(is_wh.iter().all(|&n| n <= 1));
    crate::continuation::verify_continuation(&elf_bytes, &bundle, &opts)
        .expect("verify_continuation errored")
        .expect("a keccak bundle must verify");
}

/// The recursion guest's verify path (mirrored on the host) on an inner proof
/// that contains IS_WH: the blob carries `is_wh = 1` and still verifies and
/// attests.
#[test]
fn test_recursion_blob_with_is_wh_verifies_on_host() {
    use crate::recursion::{self, MIN_PROOF_OPTIONS};

    let elf_bytes = asm_elf_bytes("test_keccak_64k");
    let inner = crate::prove_with_options_and_inputs(
        &elf_bytes,
        &[],
        &MIN_PROOF_OPTIONS,
        &MaxRowsConfig::default(),
    )
    .expect("inner prove");
    assert_eq!(inner.table_counts.is_wh, 1);

    let blob =
        recursion::encode_guest_input(&inner, &elf_bytes, &MIN_PROOF_OPTIONS).expect("encode");
    let attestation = recursion::verify_and_attest_blob(&blob, &MIN_PROOF_OPTIONS)
        .expect("verify_and_attest_blob errored")
        .expect("an inner proof with IS_WH must verify on the guest path");
    let output = recursion::check_attestation(&attestation, &elf_bytes, &MIN_PROOF_OPTIONS)
        .expect("check_attestation errored")
        .expect("attestation must bind to the inner ELF");
    assert_eq!(output, inner.public_output);
}
