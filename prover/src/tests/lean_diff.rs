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
//! Run with `cargo test -p lambda-vm-prover lean_diff -- --nocapture`.

use std::collections::HashMap;

use serde_json::Value;
use stark::constraint_ir::eval_program_base;
use stark::constraints::builder::{CaptureBuilder, ConstraintSet, num_base_from_meta};
use stark::lookup::{BusInteraction, LinearTerm, Multiplicity};

use crate::tables::types::{FE, GoldilocksExtension, GoldilocksField};

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
        "add" => Expr::Add(Box::new(parse_expr(&v["lhs"])), Box::new(parse_expr(&v["rhs"]))),
        "mul" => Expr::Mul(Box::new(parse_expr(&v["lhs"])), Box::new(parse_expr(&v["rhs"]))),
        other => panic!("unknown expression node {other}"),
    }
}

fn eval(e: &Expr, row: &[FE]) -> FE {
    match e {
        Expr::Var(i) => row[*i].clone(),
        Expr::Const(c) => FE::from(*c),
        Expr::Add(a, b) => eval(a, row) + eval(b, row),
        Expr::Mul(a, b) => eval(a, row) * eval(b, row),
    }
}

struct LeanInteract {
    channel: String,
    mult: Expr,
    msg: Vec<Expr>,
}

struct LeanCircuit {
    columns: usize,
    asserts: Vec<Expr>,
    interacts: Vec<LeanInteract>,
}

fn load_lean(name: &str) -> LeanCircuit {
    let path = format!("{}/../formal/export/{name}.json", env!("CARGO_MANIFEST_DIR"));
    let text = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{path}: {e}"));
    let json: Value = serde_json::from_str(&text).expect("json");
    let columns = json["columns"].as_u64().expect("columns") as usize;
    let mut asserts = Vec::new();
    let mut interacts = Vec::new();
    for op in json["operations"].as_array().expect("operations") {
        if let Some(a) = op.get("assert") {
            asserts.push(parse_expr(a));
        } else if let Some(i) = op.get("interact") {
            interacts.push(LeanInteract {
                channel: i["channel"].as_str().expect("channel").to_string(),
                mult: parse_expr(&i["multiplicity"]),
                msg: i["message"].as_array().expect("message").iter().map(parse_expr).collect(),
            });
        } else {
            panic!("unexpected operation {op}");
        }
    }
    LeanCircuit { columns, asserts, interacts }
}

/// Clean channel name → Lambda VM `BusId` (see `tables/types.rs`).
fn bus_id_of(channel: &str) -> u64 {
    match channel {
        "ARE_BYTES" => 0,
        "IS_HALF" => 1,
        "IS_B20" => 2,
        "MSB8" => 6,
        "MSB16" => 7,
        "ZERO" => 8,
        "HWSL" => 9,
        "BYTE_ALU" => 24,
        "ALU" => 25,
        other => panic!("unknown channel {other}"),
    }
}

// =============================================================================
// Rust side: constraints and interactions evaluated on a row
// =============================================================================

fn multiplicity_at(m: &Multiplicity, row: &[FE]) -> FE {
    let col = |c: usize| row[c].clone();
    match m {
        Multiplicity::One => FE::one(),
        Multiplicity::Column(c) => col(*c),
        Multiplicity::Sum(a, b) => col(*a) + col(*b),
        Multiplicity::Negated(c) => FE::one() - col(*c),
        Multiplicity::Diff(a, b) => col(*a) - col(*b),
        Multiplicity::Sum3(a, b, c) => col(*a) + col(*b) + col(*c),
        Multiplicity::Linear(terms) => {
            let mut acc = FE::zero();
            for t in terms {
                match *t {
                    LinearTerm::Column { coefficient, column } => {
                        acc += col(column) * FE::from(coefficient)
                    }
                    LinearTerm::ColumnUnsigned { coefficient, column } => {
                        acc += col(column) * FE::from(coefficient)
                    }
                    LinearTerm::Constant(v) => acc += FE::from(v),
                }
            }
            acc
        }
    }
}

/// `(bus id, signed multiplicity, bus elements)` of a Rust interaction on a row. A sender
/// adds `+m`, a receiver `−m`; Clean's pull is `−e` and push `+e`, so the signs agree once a
/// receiver (the table that answers the lookup, Clean's push) keeps its multiplicity.
fn rust_interaction_at(i: &BusInteraction, row: &[FE]) -> (u64, FE, Vec<FE>) {
    let m = multiplicity_at(&i.multiplicity, row);
    let m = if i.is_sender { -m } else { m };
    let elems = i.values.iter().flat_map(|v| v.combine_from(|c| row[c].clone())).collect();
    (i.bus_id, m, elems)
}

fn lean_interaction_at(i: &LeanInteract, row: &[FE]) -> (u64, FE, Vec<FE>) {
    (bus_id_of(&i.channel), eval(&i.mult, row), i.msg.iter().map(|e| eval(e, row)).collect())
}

// =============================================================================
// Matching
// =============================================================================

struct SplitMix64(u64);
impl SplitMix64 {
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
}

fn random_rows(num_cols: usize, seed: u64) -> Vec<Vec<FE>> {
    let mut rng = SplitMix64(seed);
    (0..TRIALS).map(|_| (0..num_cols).map(|_| FE::from(rng.next_u64())).collect()).collect()
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
    rust_interacts_unmatched: Vec<u64>,
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
    let rows = random_rows(num_cols, 0xC1EA_0000 ^ label.len() as u64);
    let lean_fp: Vec<Vec<u64>> = lean
        .asserts
        .iter()
        .map(|e| rows.iter().map(|r| fp(&eval(e, r))).collect())
        .collect();
    let rust_fp: Vec<Vec<u64>> =
        (0..n).map(|i| rows.iter().map(|r| fp(&eval_program_base(&prog, i, r))).collect()).collect();
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
    let lean_interacts_unmatched = li.iter().map(|&i| lean.interacts[i].channel.clone()).collect();
    let rust_interacts_unmatched = ri.iter().map(|&j| interactions[j].bus_id).collect();

    // Every Lean constraint vanishes on the honest trace.
    for (r, row) in honest_rows.iter().enumerate() {
        for (i, e) in lean.asserts.iter().enumerate() {
            assert_eq!(eval(e, row), FE::zero(), "[{label}] Lean assert {i} on honest row {r}");
        }
        for i in 0..n {
            assert_eq!(eval_program_base(&prog, i, row), FE::zero(), "[{label}] Rust constraint {i} on honest row {r}");
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
        lean_asserts_unmatched, rust_constraints_unmatched, lean_interacts_unmatched, rust_interacts_unmatched
    );
    Report {
        lean_asserts_unmatched,
        rust_constraints_unmatched,
        lean_interacts_unmatched,
        rust_interacts_unmatched,
    }
}

fn rows_of(trace: &stark::trace::TraceTable<Gl, Gl3>, num_cols: usize) -> Vec<Vec<FE>> {
    (0..trace.num_rows())
        .map(|r| (0..num_cols).map(|c| trace.main_table.get(r, c).clone()).collect())
        .collect()
}

const EDGE: [u64; 8] = [0, 1, 0xFF, 0xFFFF, 0x7FFF_FFFF_FFFF_FFFF, 0x8000_0000_0000_0000, u64::MAX - 1, u64::MAX];

// =============================================================================
// Tables
// =============================================================================

#[test]
fn lean_diff_lt() {
    use crate::tables::lt::{LtConstraints, LtOperation, bus_interactions, cols, generate_lt_trace};
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
    for _ in 0..256 {
        ops.push(LtOperation::new_with_invert(
            rng.next_u64(),
            rng.next_u64(),
            rng.next_u64() & 1 == 1,
            rng.next_u64() & 1 == 1,
        ));
    }
    let trace = generate_lt_trace(&ops);
    let report = compare(
        "LT",
        &load_lean("lt"),
        &LtConstraints,
        &bus_interactions(),
        cols::NUM_COLUMNS,
        &rows_of(&trace, cols::NUM_COLUMNS),
    );
    // Known deltas: the Lean model keeps `μ·(μ − 1) = 0` (spec: `μ : Bit`), the Rust chip bounds
    // `μ` with the `ARE_BYTES[μ, 0]` send instead (PR #1016).
    assert_eq!(report.lean_asserts_unmatched.len(), 1, "[LT] only the μ bit constraint should be Lean-only");
    assert!(report.rust_constraints_unmatched.is_empty(), "[LT] Rust-only constraints");
    assert!(report.lean_interacts_unmatched.is_empty(), "[LT] Lean-only interactions");
    assert_eq!(report.rust_interacts_unmatched, vec![0], "[LT] only ARE_BYTES[μ, 0] should be Rust-only");
}

#[test]
fn lean_diff_eq() {
    use crate::tables::eq::{EqConstraints, EqOperation, bus_interactions, cols, generate_eq_trace};
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
        let b = if rng.next_u64() & 1 == 1 { a } else { rng.next_u64() };
        ops.push(EqOperation::new(a, b, rng.next_u64() & 1 == 1));
    }
    let trace = generate_eq_trace(&ops);
    let report = compare(
        "EQ",
        &load_lean("eq"),
        &EqConstraints,
        &bus_interactions(),
        cols::NUM_COLUMNS,
        &rows_of(&trace, cols::NUM_COLUMNS),
    );
    assert!(report.lean_asserts_unmatched.is_empty() && report.rust_constraints_unmatched.is_empty(), "[EQ] constraints differ");
    assert!(report.lean_interacts_unmatched.is_empty() && report.rust_interacts_unmatched.is_empty(), "[EQ] interactions differ");
}

#[test]
fn lean_diff_shift() {
    use crate::tables::shift::{ShiftConstraints, ShiftOperation, bus_interactions, cols, generate_shift_trace};
    let mut rng = SplitMix64(0x5F);
    let mut ops = Vec::new();
    for &v in &EDGE {
        for s in [0u64, 1, 15, 16, 17, 31, 32, 33, 47, 48, 63, 64, 65, 255, 256, u64::MAX] {
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
    let report = compare(
        "SHIFT",
        &load_lean("shift"),
        &ShiftConstraints,
        &bus_interactions(),
        cols::NUM_COLUMNS,
        &rows_of(&trace, cols::NUM_COLUMNS),
    );
    assert!(report.lean_asserts_unmatched.is_empty() && report.rust_constraints_unmatched.is_empty(), "[SHIFT] constraints differ");
    assert!(report.lean_interacts_unmatched.is_empty() && report.rust_interacts_unmatched.is_empty(), "[SHIFT] interactions differ");
}
