//! Differential check of the Lean models in `formal/` against the Rust tables.
//!
//! `formal/export/<table>.json` holds the operations of each Lean row circuit:
//! the constraint polynomials (`assert`) and the bus interactions (`interact`),
//! with variables numbered in Rust column order. For each of LT, EQ and SHIFT
//! this test evaluates them on random rows next to the Rust `ConstraintSet` and
//! `bus_interactions()`, and matches them as multisets of functions
//! (Schwartz–Zippel over [`TRIALS`] rows). It also checks that the Lean
//! constraints vanish on honest Rust traces, padding rows included.
//!
//! `formal/export/bitwise.json` holds the size of the Lean BITWISE table and of
//! each static lookup channel, and a sample of the table: the precomputed
//! columns of sampled rows and the message each channel serves there, checked
//! against `generate_bitwise_row` and the receivers of
//! `bitwise::bus_interactions()`. Every one of the 2^20 rows is also checked
//! against a transcription of the Lean row function, pinned to the Lean export
//! on the sampled rows.
//!
//! Run with `cargo test -p lambda-vm-prover lean_diff -- --nocapture`.

use std::collections::HashMap;

use serde_json::Value;
use stark::constraint_ir::eval_program_base;
use stark::constraints::builder::{CaptureBuilder, ConstraintSet, num_base_from_meta};
use stark::lookup::{BusInteraction, Multiplicity};

use super::test_rng::SplitMix64;
use crate::tables::types::{BusId, FE, GoldilocksExtension, GoldilocksField};

type Gl = GoldilocksField;
type Gl3 = GoldilocksExtension;

const TRIALS: usize = 64;

// =============================================================================
// Lean side: the exported operations
// =============================================================================

enum Expr {
    Var(usize),
    Const(u64),
    Add(Box<Expr>, Box<Expr>),
    Mul(Box<Expr>, Box<Expr>),
}

fn parse_expr(v: &Value) -> Expr {
    match v["type"].as_str().expect("expr type") {
        "var" => Expr::Var(v["index"].as_u64().expect("var index") as usize),
        "const" => Expr::Const(v["value"].as_u64().expect("const value")),
        "add" => Expr::Add(
            Box::new(parse_expr(&v["lhs"])),
            Box::new(parse_expr(&v["rhs"])),
        ),
        "mul" => Expr::Mul(
            Box::new(parse_expr(&v["lhs"])),
            Box::new(parse_expr(&v["rhs"])),
        ),
        other => panic!("unknown expression node {other}"),
    }
}

fn eval(e: &Expr, row: &[FE]) -> FE {
    match e {
        Expr::Var(i) => row[*i],
        Expr::Const(c) => FE::from(*c),
        Expr::Add(a, b) => eval(a, row) + eval(b, row),
        Expr::Mul(a, b) => eval(a, row) * eval(b, row),
    }
}

struct LeanInteract {
    channel: String,
    /// Clean's direction: a pull takes the message from the channel, a push adds it.
    pull: bool,
    mult: Expr,
    msg: Vec<Expr>,
}

struct LeanCircuit {
    columns: usize,
    asserts: Vec<Expr>,
    interacts: Vec<LeanInteract>,
}

fn load_export(name: &str) -> Value {
    let path = format!(
        "{}/../formal/export/{name}.json",
        env!("CARGO_MANIFEST_DIR")
    );
    let text = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{path}: {e}"));
    serde_json::from_str(&text).expect("json")
}

fn load_lean(name: &str) -> LeanCircuit {
    let json = load_export(name);
    let columns = json["columns"].as_u64().expect("columns") as usize;
    let mut asserts = Vec::new();
    let mut interacts = Vec::new();
    for op in json["operations"].as_array().expect("operations") {
        if let Some(a) = op.get("assert") {
            asserts.push(parse_expr(a));
        } else if let Some(i) = op.get("interact") {
            interacts.push(LeanInteract {
                channel: i["channel"].as_str().expect("channel").to_string(),
                pull: match i["direction"].as_str().expect("direction") {
                    "pull" => true,
                    "push" => false,
                    other => panic!("unknown direction {other}"),
                },
                mult: parse_expr(&i["multiplicity"]),
                msg: i["message"]
                    .as_array()
                    .expect("message")
                    .iter()
                    .map(parse_expr)
                    .collect(),
            });
        } else {
            panic!("unexpected operation {op}");
        }
    }
    LeanCircuit {
        columns,
        asserts,
        interacts,
    }
}

/// Clean channel name → Lambda VM `BusId` (see `tables/types.rs`).
fn bus_id_of(channel: &str) -> u64 {
    let bus = match channel {
        "ARE_BYTES" => BusId::AreBytes,
        "IS_HALF" => BusId::IsHalfword,
        "IS_B20" => BusId::IsB20,
        "MSB8" => BusId::Msb8,
        "MSB16" => BusId::Msb16,
        "ZERO" => BusId::Zero,
        "HWSL" => BusId::Hwsl,
        "BYTE_ALU" => BusId::ByteAlu,
        "ALU" => BusId::Alu,
        other => panic!("unknown channel {other}"),
    };
    bus as u64
}

/// Padding rows are the rows with `μ = 0`; the honest-row checks cover them only if the
/// trace has some.
fn assert_has_padding(label: &str, honest: &[Vec<FE>], mu: usize) {
    assert!(
        honest.iter().any(|r| r[mu] == FE::zero()),
        "[{label}] the trace has padding rows"
    );
}

// =============================================================================
// Rust side: constraints and interactions evaluated on a row
// =============================================================================

/// `(bus id, signed multiplicity, bus elements)` of a Rust interaction on a row, signed the way
/// Clean signs it: a sender (Clean's pull) counts `−m`, a receiver (the table that answers the
/// lookup, Clean's push) counts `+m`. This is the opposite of the prover's LogUp sum, where a
/// sender adds `+m`; `lean_interaction_at` signs Lean's pulls the same way.
fn rust_interaction_at(i: &BusInteraction, row: &[FE]) -> (u64, FE, Vec<FE>) {
    let m = i.multiplicity.evaluate_with(|c| row[c]);
    let m = if i.is_sender { -m } else { m };
    let elems = i
        .values
        .iter()
        .flat_map(|v| v.combine_from(|c| row[c]))
        .collect();
    (i.bus_id, m, elems)
}

fn lean_interaction_at(i: &LeanInteract, row: &[FE]) -> (u64, FE, Vec<FE>) {
    let m = eval(&i.mult, row);
    (
        bus_id_of(&i.channel),
        if i.pull { -m } else { m },
        i.msg.iter().map(|e| eval(e, row)).collect(),
    )
}

// =============================================================================
// Matching
// =============================================================================

/// FNV-1a of the label: a seed that differs between tables.
fn seed_of(label: &str) -> u64 {
    label.bytes().fold(0xCBF2_9CE4_8422_2325, |h, b| {
        (h ^ u64::from(b)).wrapping_mul(0x0100_0000_01B3)
    })
}

fn random_rows(num_cols: usize, seed: u64) -> Vec<Vec<FE>> {
    let mut rng = SplitMix64(seed);
    (0..TRIALS)
        .map(|_| (0..num_cols).map(|_| FE::from(rng.next_u64())).collect())
        .collect()
}

/// Greedy multiset matching of fingerprints; returns the unmatched indices on each side.
fn unmatched<K: std::hash::Hash + Eq + Clone>(lean: &[K], rust: &[K]) -> (Vec<usize>, Vec<usize>) {
    let mut pool: HashMap<K, Vec<usize>> = HashMap::new();
    for (j, k) in rust.iter().enumerate() {
        pool.entry(k.clone()).or_default().push(j);
    }
    let mut lean_unmatched = Vec::new();
    for (i, k) in lean.iter().enumerate() {
        match pool.get_mut(k).and_then(|v| v.pop()) {
            Some(_) => {}
            None => lean_unmatched.push(i),
        }
    }
    let rust_unmatched: Vec<usize> = pool.into_values().flatten().collect();
    (lean_unmatched, rust_unmatched)
}

fn fp(x: &FE) -> u64 {
    x.canonical_u64()
}

struct Report {
    lean_asserts_unmatched: Vec<usize>,
    rust_constraints_unmatched: Vec<usize>,
    lean_interacts_unmatched: Vec<String>,
    /// Indices into the Rust interactions.
    rust_interacts_unmatched: Vec<usize>,
    /// The random rows the fingerprints were taken on.
    random_rows: Vec<Vec<FE>>,
}

fn compare<CS: ConstraintSet<Gl, Gl3>>(
    label: &str,
    lean: &LeanCircuit,
    set: &CS,
    interactions: &[BusInteraction],
    num_cols: usize,
    honest_rows: &[Vec<FE>],
) -> Report {
    assert_eq!(lean.columns, num_cols, "[{label}] column count");

    // The Rust constraints as an evaluable program.
    let meta = set.meta();
    let n = meta.len();
    let mut cb = CaptureBuilder::<Gl, Gl3>::new();
    set.eval(&mut cb);
    let (prog, _) = cb.finish(num_base_from_meta(&meta));

    // Fingerprints over random rows.
    let rows = random_rows(num_cols, seed_of(label));
    let lean_fp: Vec<Vec<u64>> = lean
        .asserts
        .iter()
        .map(|e| rows.iter().map(|r| fp(&eval(e, r))).collect())
        .collect();
    let rust_fp: Vec<Vec<u64>> = (0..n)
        .map(|i| {
            rows.iter()
                .map(|r| fp(&eval_program_base(&prog, i, r)))
                .collect()
        })
        .collect();
    let (lean_asserts_unmatched, rust_constraints_unmatched) = unmatched(&lean_fp, &rust_fp);

    let lean_ifp: Vec<(u64, Vec<u64>, Vec<Vec<u64>>)> = lean
        .interacts
        .iter()
        .map(|i| {
            let evals: Vec<_> = rows.iter().map(|r| lean_interaction_at(i, r)).collect();
            (
                evals[0].0,
                evals.iter().map(|e| fp(&e.1)).collect(),
                evals.iter().map(|e| e.2.iter().map(fp).collect()).collect(),
            )
        })
        .collect();
    let rust_ifp: Vec<(u64, Vec<u64>, Vec<Vec<u64>>)> = interactions
        .iter()
        .map(|i| {
            let evals: Vec<_> = rows.iter().map(|r| rust_interaction_at(i, r)).collect();
            (
                evals[0].0,
                evals.iter().map(|e| fp(&e.1)).collect(),
                evals.iter().map(|e| e.2.iter().map(fp).collect()).collect(),
            )
        })
        .collect();
    let (li, ri) = unmatched(&lean_ifp, &rust_ifp);
    let lean_interacts_unmatched = li
        .iter()
        .map(|&i| lean.interacts[i].channel.clone())
        .collect();
    let mut rust_interacts_unmatched = ri;
    rust_interacts_unmatched.sort_unstable();

    // Every Lean constraint with a Rust counterpart vanishes on the honest trace. The
    // Lean-only ones are the documented deltas (LT's `μ·(1 − μ)` does not hold on a
    // deduplicated Rust row with `μ > 1`).
    for (r, row) in honest_rows.iter().enumerate() {
        for (i, e) in lean.asserts.iter().enumerate() {
            if lean_asserts_unmatched.contains(&i) {
                continue;
            }
            assert_eq!(
                eval(e, row),
                FE::zero(),
                "[{label}] Lean assert {i} on honest row {r}"
            );
        }
        for i in 0..n {
            assert_eq!(
                eval_program_base(&prog, i, row),
                FE::zero(),
                "[{label}] Rust constraint {i} on honest row {r}"
            );
        }
    }

    println!(
        "[{label}] Lean: {} asserts, {} interactions | Rust: {} constraints, {} interactions",
        lean.asserts.len(),
        lean.interacts.len(),
        n,
        interactions.len()
    );
    println!(
        "[{label}] unmatched Lean asserts {:?}, unmatched Rust constraints {:?}, unmatched Lean interactions {:?}, unmatched Rust interactions (bus ids) {:?}",
        lean_asserts_unmatched,
        rust_constraints_unmatched,
        lean_interacts_unmatched,
        rust_interacts_unmatched
            .iter()
            .map(|&j| interactions[j].bus_id)
            .collect::<Vec<_>>()
    );
    Report {
        lean_asserts_unmatched,
        rust_constraints_unmatched,
        lean_interacts_unmatched,
        rust_interacts_unmatched,
        random_rows: rows,
    }
}

fn rows_of(trace: &stark::trace::TraceTable<Gl, Gl3>, num_cols: usize) -> Vec<Vec<FE>> {
    (0..trace.num_rows())
        .map(|r| trace.main_table.get_row(r)[..num_cols].to_vec())
        .collect()
}

const EDGE: [u64; 8] = [
    0,
    1,
    0xFF,
    0xFFFF,
    0x7FFF_FFFF_FFFF_FFFF,
    0x8000_0000_0000_0000,
    u64::MAX - 1,
    u64::MAX,
];

// =============================================================================
// Tables
// =============================================================================

#[test]
fn lean_diff_lt() {
    use crate::tables::lt::{
        LtConstraints, LtOperation, bus_interactions, cols, generate_lt_trace,
    };
    let mut rng = SplitMix64(0x17);
    let mut ops = Vec::new();
    for &a in &EDGE {
        for &b in &EDGE {
            for signed in [false, true] {
                for invert in [false, true] {
                    ops.push(LtOperation::new_with_invert(a, b, signed, invert));
                }
            }
        }
    }
    for _ in 0..255 {
        ops.push(LtOperation::new_with_invert(
            rng.next_u64(),
            rng.next_u64(),
            rng.next_u64() & 1 == 1,
            rng.next_u64() & 1 == 1,
        ));
    }
    // Repeated operations: deduplicated rows with μ = 3, and one op split into μ = MU_MAX and
    // μ = 2 rows. 514 distinct rows pad to 1024, so padding rows are checked too.
    for _ in 0..3 {
        ops.push(LtOperation::new_with_invert(5, 7, false, false));
    }
    for _ in 0..crate::tables::types::MU_MAX + 2 {
        ops.push(LtOperation::new_with_invert(1, 2, true, true));
    }
    let trace = generate_lt_trace(&ops);
    let lean = load_lean("lt");
    let interactions = bus_interactions();
    let honest = rows_of(&trace, cols::NUM_COLUMNS);
    assert_has_padding("LT", &honest, cols::MU);
    let report = compare(
        "LT",
        &lean,
        &LtConstraints,
        &interactions,
        cols::NUM_COLUMNS,
        &honest,
    );
    // Known deltas: the Lean model keeps `μ·(1 − μ) = 0` (assert 0), the Rust chip bounds
    // `μ` with the `ARE_BYTES[μ, 0]` send instead. Both are pinned by content, not position.
    assert_eq!(
        report.lean_asserts_unmatched,
        vec![0],
        "[LT] only the μ bit constraint should be Lean-only"
    );
    let mu_bit = |r: &[FE]| r[cols::MU] * (FE::one() - r[cols::MU]);
    for r in &report.random_rows {
        assert_eq!(
            eval(&lean.asserts[0], r),
            mu_bit(r),
            "[LT] Lean-only assert is μ·(1 − μ)"
        );
    }
    // The delta is real on honest rows: it fails exactly on the deduplicated rows.
    let mut dedup_rows = 0;
    for r in &honest {
        let mu = r[cols::MU].canonical_u64();
        assert_eq!(
            eval(&lean.asserts[0], r) == FE::zero(),
            mu <= 1,
            "[LT] μ = {mu}"
        );
        dedup_rows += usize::from(mu > 1);
    }
    assert!(dedup_rows >= 2, "[LT] the trace has deduplicated rows");
    assert!(
        report.rust_constraints_unmatched.is_empty(),
        "[LT] Rust-only constraints"
    );
    assert!(
        report.lean_interacts_unmatched.is_empty(),
        "[LT] Lean-only interactions"
    );
    assert_eq!(
        report.rust_interacts_unmatched.len(),
        1,
        "[LT] only ARE_BYTES[μ, 0] should be Rust-only"
    );
    let are_bytes = &interactions[report.rust_interacts_unmatched[0]];
    assert!(
        are_bytes.is_sender && matches!(are_bytes.multiplicity, Multiplicity::One),
        "[LT] the μ bound is sent with multiplicity One"
    );
    for r in &report.random_rows {
        assert_eq!(
            rust_interaction_at(are_bytes, r),
            (
                BusId::AreBytes as u64,
                -FE::one(),
                vec![r[cols::MU], FE::zero()]
            ),
            "[LT] the Rust-only interaction is ARE_BYTES[μ, 0]"
        );
    }
}

#[test]
fn lean_diff_eq() {
    use crate::tables::eq::{
        EqConstraints, EqOperation, bus_interactions, cols, generate_eq_trace,
    };
    let mut rng = SplitMix64(0xE0);
    let mut ops = Vec::new();
    for &a in &EDGE {
        for &b in &EDGE {
            for invert in [false, true] {
                ops.push(EqOperation::new(a, b, invert));
            }
        }
    }
    for _ in 0..256 {
        let a = rng.next_u64();
        let b = if rng.next_u64() & 1 == 1 {
            a
        } else {
            rng.next_u64()
        };
        ops.push(EqOperation::new(a, b, rng.next_u64() & 1 == 1));
    }
    let trace = generate_eq_trace(&ops);
    let honest = rows_of(&trace, cols::NUM_COLUMNS);
    assert_has_padding("EQ", &honest, cols::MU);
    let report = compare(
        "EQ",
        &load_lean("eq"),
        &EqConstraints,
        &bus_interactions(),
        cols::NUM_COLUMNS,
        &honest,
    );
    assert!(
        report.lean_asserts_unmatched.is_empty() && report.rust_constraints_unmatched.is_empty(),
        "[EQ] constraints differ"
    );
    assert!(
        report.lean_interacts_unmatched.is_empty() && report.rust_interacts_unmatched.is_empty(),
        "[EQ] interactions differ"
    );
}

#[test]
fn lean_diff_shift() {
    use crate::tables::shift::{
        ShiftConstraints, ShiftOperation, bus_interactions, cols, generate_shift_trace,
    };
    let mut rng = SplitMix64(0x5F);
    let mut ops = Vec::new();
    for &v in &EDGE {
        for s in [
            0u64,
            1,
            15,
            16,
            17,
            31,
            32,
            33,
            47,
            48,
            63,
            64,
            65,
            255,
            256,
            u64::MAX,
        ] {
            for direction in [false, true] {
                for signed in [false, true] {
                    for word in [false, true] {
                        ops.push(ShiftOperation::new(v, s, direction, signed, word));
                    }
                }
            }
        }
    }
    for _ in 0..512 {
        ops.push(ShiftOperation::new(
            rng.next_u64(),
            rng.next_u64(),
            rng.next_u64() & 1 == 1,
            rng.next_u64() & 1 == 1,
            rng.next_u64() & 1 == 1,
        ));
    }
    let trace = generate_shift_trace(&ops);
    let honest = rows_of(&trace, cols::NUM_COLUMNS);
    assert_has_padding("SHIFT", &honest, cols::MU);
    let report = compare(
        "SHIFT",
        &load_lean("shift"),
        &ShiftConstraints,
        &bus_interactions(),
        cols::NUM_COLUMNS,
        &honest,
    );
    assert!(
        report.lean_asserts_unmatched.is_empty() && report.rust_constraints_unmatched.is_empty(),
        "[SHIFT] constraints differ"
    );
    assert!(
        report.lean_interacts_unmatched.is_empty() && report.rust_interacts_unmatched.is_empty(),
        "[SHIFT] interactions differ"
    );
}

/// `Bitwise.row i` of `formal/LambdaVM/Tables/Bitwise.lean`, transcribed: `rowOf` on
/// `i % 2^8`, `i / 2^8 % 2^8`, `i / 2^16 % 16`, with the Lean `sll`/`sllc` and their `z = 0` case.
fn lean_bitwise_row(i: usize) -> [u64; 11] {
    let (x, y, z) = (
        (i % (1 << 8)) as u64,
        (i / (1 << 8) % (1 << 8)) as u64,
        (i / (1 << 16) % 16) as u64,
    );
    let h = x + (1 << 8) * y;
    let sll = if z == 0 { h } else { (h << z) % (1 << 16) };
    let sllc = if z == 0 { 0 } else { h >> (16 - z) };
    let zero = u64::from(x == 0 && y == 0 && z == 0);
    [
        x,
        y,
        z,
        x & y,
        x | y,
        x ^ y,
        x / (1 << 7),
        h / (1 << 15),
        zero,
        sll,
        sllc,
    ]
}

#[test]
fn lean_diff_bitwise() {
    use crate::tables::bitwise::{
        NUM_PRECOMPUTED_COLS, NUM_ROWS, bus_interactions, cols, generate_bitwise_row,
    };
    let json = load_export("bitwise");
    let as_u64s = |v: &Value| -> Vec<u64> {
        v.as_array()
            .expect("array")
            .iter()
            .map(|x| x.as_u64().expect("number"))
            .collect()
    };
    let indices: Vec<usize> = as_u64s(&json["indices"])
        .into_iter()
        .map(|i| i as usize)
        .collect();
    let lean_rows = json["rows"].as_array().expect("rows");
    assert_eq!(
        lean_rows.len(),
        indices.len(),
        "[BITWISE] one row per index"
    );
    assert_eq!(
        json["numRows"].as_u64().expect("numRows") as usize,
        NUM_ROWS,
        "[BITWISE] table size"
    );

    // All 2^20 rows: Rust against the transcribed Lean row function, which the sampled
    // rows below pin to the Lean export.
    for i in 0..NUM_ROWS {
        assert_eq!(
            generate_bitwise_row(i),
            lean_bitwise_row(i),
            "[BITWISE] row {i}"
        );
    }

    // The precomputed columns of every sampled row.
    let rust_rows: Vec<Vec<FE>> = indices
        .iter()
        .zip(lean_rows)
        .map(|(&i, lean)| {
            let rust = generate_bitwise_row(i);
            assert_eq!(as_u64s(lean), rust.to_vec(), "[BITWISE] row {i}");
            assert_eq!(
                as_u64s(lean),
                lean_bitwise_row(i).to_vec(),
                "[BITWISE] transcription of the Lean row function, row {i}"
            );
            let mut row = vec![FE::zero(); cols::NUM_COLUMNS];
            for (c, v) in rust.iter().enumerate().take(NUM_PRECOMPUTED_COLS) {
                row[c] = FE::from(*v);
            }
            row
        })
        .collect();

    // Every Lean channel is a bus with BITWISE receivers serving the same messages, and
    // every BITWISE receiver belongs to a Lean channel.
    let interactions = bus_interactions();
    let mut receivers: HashMap<u64, Vec<&BusInteraction>> = HashMap::new();
    for i in &interactions {
        assert!(!i.is_sender, "[BITWISE] only receives");
        receivers.entry(i.bus_id).or_default().push(i);
    }
    let channels = json["channels"].as_array().expect("channels");
    let mut seen: Vec<u64> = Vec::new();
    for ch in channels {
        let name = ch["channel"].as_str().expect("channel");
        let bus = bus_id_of(name);
        assert!(
            !seen.contains(&bus),
            "[BITWISE] channel {name} exported twice"
        );
        seen.push(bus);
        let rust = receivers
            .get(&bus)
            .unwrap_or_else(|| panic!("[BITWISE] no Rust receiver for {name}"));
        // One table entry per row and receiver: BYTE_ALU serves three per row.
        assert_eq!(
            ch["size"].as_u64().expect("size") as usize,
            NUM_ROWS * rust.len(),
            "[BITWISE] size of {name}"
        );
        let messages = ch["messages"].as_array().expect("messages");
        assert_eq!(
            messages.len(),
            indices.len(),
            "[BITWISE] {name}: one message list per sampled row"
        );
        for ((&i, row), lean) in indices.iter().zip(&rust_rows).zip(messages) {
            let mut lean: Vec<Vec<u64>> = lean
                .as_array()
                .expect("messages of a row")
                .iter()
                .map(as_u64s)
                .collect();
            let mut rust: Vec<Vec<u64>> = rust
                .iter()
                .map(|r| {
                    r.values
                        .iter()
                        .flat_map(|v| v.combine_from(|c| row[c]))
                        .map(|e| e.canonical_u64())
                        .collect()
                })
                .collect();
            lean.sort();
            rust.sort();
            assert_eq!(lean, rust, "[BITWISE] {name} at row {i}");
        }
    }
    let mut rust_buses: Vec<u64> = receivers.keys().copied().collect();
    rust_buses.sort_unstable();
    seen.sort_unstable();
    assert_eq!(
        seen, rust_buses,
        "[BITWISE] Lean channels vs Rust receivers"
    );
    println!(
        "[BITWISE] {NUM_ROWS} rows against the row function, {} sampled rows, {} channels, {} receivers: exact match",
        indices.len(),
        seen.len(),
        interactions.len()
    );
}
