//! The canonical row order of the six deduplicated tables
//! ([`crate::tables::row_order`]): LT, BRANCH, BYTEWISE, EQ, MUL and DVRM.
//!
//! Every test here reads the order the process runs in
//! ([`canonical_rows`]) and asserts what that order must give:
//! - sorted (the default): a table is a function of its operations' multiset,
//!   two builds of a run's traces are equal, and two proves of a run give the
//!   same bytes;
//! - `LAMBDA_VM_CANONICAL_ROWS=0` (`HashMap` order, the control): the same
//!   comparisons DIFFER, which is what makes the default's equalities mean
//!   something. Run the tests once in each setting.
//!
//! The run is `dedup_rows_64.s`: every one of the six tables gets about one
//! unique row per loop iteration, so two `HashMap` orders of it cannot
//! coincide by chance. The run-level tests are `#[ignore]` (box): they build
//! the 2^20-row BITWISE table.

use executor::elf::Elf;
use executor::vm::execution::Executor;
use stark::proof::options::{GoldilocksCubicProofOptions, ProofOptions};
use stark::table::Table;

use crate::VmProof;
use crate::tables::row_order::canonical_rows;
use crate::tables::trace_builder::Traces;
use crate::tables::types::{GoldilocksExtension, GoldilocksField, alu_op};
use crate::tables::{MaxRowsConfig, branch, bytewise, dvrm, eq, lt, mul};
use crate::test_utils::asm_elf_bytes;

/// The program whose six dedup tables are each at least 64 rows wide.
const RUN: &str = "dedup_rows_64";

/// The order this process runs in, for messages.
fn order() -> &'static str {
    if canonical_rows() {
        "sorted (the default)"
    } else {
        "HashMap order (LAMBDA_VM_CANONICAL_ROWS=0, the control)"
    }
}

/// splitmix64: a fixed stream of test operands.
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    fn below(&mut self, n: usize) -> usize {
        (self.next() % n as u64) as usize
    }

    /// `count` draws from `distinct` distinct values, so most repeat.
    fn with_repeats<T: Clone>(
        &mut self,
        distinct: usize,
        count: usize,
        f: impl Fn(&mut Self) -> T,
    ) -> Vec<T> {
        let pool: Vec<T> = (0..distinct).map(|_| f(self)).collect();
        (0..count)
            .map(|_| pool[self.below(distinct)].clone())
            .collect()
    }

    fn shuffled<T: Clone>(&mut self, xs: &[T]) -> Vec<T> {
        let mut out = xs.to_vec();
        for i in (1..out.len()).rev() {
            let j = self.below(i + 1);
            out.swap(i, j);
        }
        out
    }
}

type Main = Table<GoldilocksField>;

/// One dedup table's generator over an operation list (`build(false)`) or the
/// same list shuffled (`build(true)`).
struct Case {
    name: &'static str,
    build: Box<dyn Fn(bool) -> Main>,
}

/// The six generators, each over 400 operations drawn from 150 distinct ones.
fn cases() -> Vec<Case> {
    let mut rng = Rng(0x0D0_5EED);
    let case = |name: &'static str, build: Box<dyn Fn(bool) -> Main>| Case { name, build };
    let lts = rng.with_repeats(150, 400, |r| {
        lt::LtOperation::new_with_invert(r.next(), r.next(), r.next() & 1 == 1, r.next() & 1 == 1)
    });
    let branches = rng.with_repeats(150, 400, |r| {
        branch::BranchOperation::new(r.next() & !3, r.next() & 0xFFE, r.next(), r.next() & 1 == 1)
    });
    let bytewises = rng.with_repeats(150, 400, |r| {
        let op = [alu_op::AND, alu_op::OR, alu_op::XOR][r.below(3)];
        bytewise::BytewiseOperation::new(r.next(), r.next(), op)
    });
    let eqs = rng.with_repeats(150, 400, |r| {
        let a = r.next();
        let b = if r.next() & 1 == 1 { a } else { r.next() };
        eq::EqOperation::new(a, b, r.next() & 1 == 1)
    });
    let muls = rng.with_repeats(150, 400, |r| {
        (
            mul::MulOperation::new(r.next(), r.next() & 1 == 1, r.next(), r.next() & 1 == 1),
            r.next() & 1 == 1,
        )
    });
    let dvrms = rng.with_repeats(150, 400, |r| {
        (
            dvrm::DvrmOperation::new(r.next(), r.next() | 1, r.next() & 1 == 1),
            r.next() & 1 == 1,
        )
    });
    let (lts2, branches2, bytewises2, eqs2, muls2, dvrms2) = (
        rng.shuffled(&lts),
        rng.shuffled(&branches),
        rng.shuffled(&bytewises),
        rng.shuffled(&eqs),
        rng.shuffled(&muls),
        rng.shuffled(&dvrms),
    );
    vec![
        case(
            "LT",
            Box::new(move |s| lt::generate_lt_trace(if s { &lts2 } else { &lts }).main_table),
        ),
        case(
            "BRANCH",
            Box::new(move |s| {
                branch::generate_branch_trace(if s { &branches2 } else { &branches }).main_table
            }),
        ),
        case(
            "BYTEWISE",
            Box::new(move |s| {
                bytewise::generate_bytewise_trace(if s { &bytewises2 } else { &bytewises })
                    .main_table
            }),
        ),
        case(
            "EQ",
            Box::new(move |s| eq::generate_eq_trace(if s { &eqs2 } else { &eqs }).main_table),
        ),
        case(
            "MUL",
            Box::new(move |s| mul::generate_mul_trace(if s { &muls2 } else { &muls }).main_table),
        ),
        case(
            "DVRM",
            Box::new(move |s| {
                dvrm::generate_dvrm_trace(if s { &dvrms2 } else { &dvrms }).main_table
            }),
        ),
    ]
}

/// A table's rows as canonical integers, sorted: its multiset of rows.
fn row_multiset(t: &Main) -> Vec<Vec<u64>> {
    let mut rows: Vec<Vec<u64>> = (0..t.height)
        .map(|i| t.get_row(i).iter().map(|v| v.canonical()).collect())
        .collect();
    rows.sort();
    rows
}

/// ★ Sorted: a dedup table is a function of its operations' multiset, so the
/// same operations in another order give the same table, cell for cell. In
/// `HashMap` order (the control) they give another table: another map, another
/// iteration order. Either way the two tables hold the same rows.
#[test]
fn each_dedup_table_is_a_function_of_its_operations() {
    for c in cases() {
        let (drawn, shuffled) = ((c.build)(false), (c.build)(true));
        assert!(drawn.height >= 128, "{}: {} rows", c.name, drawn.height);
        assert_eq!(
            row_multiset(&drawn),
            row_multiset(&shuffled),
            "{}: the same operations gave other rows",
            c.name
        );
        assert_eq!(
            drawn == shuffled,
            canonical_rows(),
            "{} in {}: the two builds are {}",
            c.name,
            order(),
            if drawn == shuffled {
                "equal"
            } else {
                "different"
            }
        );
    }
}

/// No generator of the six lays out its rows in map order: each hands its map
/// to [`crate::tables::row_order::unique_rows`] and none iterates the map itself.
#[test]
fn the_six_generators_take_their_row_order_from_row_order() {
    for (name, src) in [
        ("lt.rs", include_str!("../tables/lt.rs")),
        ("branch.rs", include_str!("../tables/branch.rs")),
        ("bytewise.rs", include_str!("../tables/bytewise.rs")),
        ("eq.rs", include_str!("../tables/eq.rs")),
        ("mul.rs", include_str!("../tables/mul.rs")),
        ("dvrm.rs", include_str!("../tables/dvrm.rs")),
    ] {
        assert_eq!(
            src.matches("super::row_order::unique_rows(op_map)").count(),
            1,
            "{name}: its rows must come from row_order::unique_rows"
        );
        for iteration in [
            "op_map.into_iter",
            "op_map.iter",
            "op_map.drain",
            "op_map.keys",
            "op_map.values",
            "in op_map",
            "in &op_map",
        ] {
            assert!(
                !src.contains(iteration),
                "{name} iterates its map: `{iteration}`"
            );
        }
    }
}

/// `dedup_rows_64.s` runs and halts: 5 setup instructions, 64 iterations of 14
/// (`beq` never branches, so `add` runs every time; the last `blt` falls
/// through), and 3 to halt.
#[test]
fn the_dedup_run_executes() {
    let elf = Elf::load(&asm_elf_bytes(RUN)).expect("load the ELF");
    let run = Executor::new(&elf, vec![])
        .expect("executor")
        .run()
        .expect("run");
    assert_eq!(run.logs.len(), 5 + 64 * 14 + 3, "cycles");
}

/// The traces of one execution of `name`.
fn build_traces(name: &str) -> Traces {
    let elf = Elf::load(&asm_elf_bytes(name)).expect("load the ELF");
    let run = Executor::new(&elf, vec![])
        .expect("executor")
        .run()
        .expect("run");
    Traces::from_elf_and_logs(
        &elf,
        &run.logs,
        &MaxRowsConfig::default(),
        &[],
        #[cfg(feature = "disk-spill")]
        stark::storage_mode::StorageMode::Ram,
    )
    .expect("build the traces")
}

type Trace = stark::trace::TraceTable<GoldilocksField, GoldilocksExtension>;

fn mains<'a>(name: &'static str, ts: &'a [Trace]) -> (&'static str, Vec<&'a Main>) {
    (name, ts.iter().map(|t| &t.main_table).collect())
}

/// The six dedup tables of a build, by name, chunk by chunk.
fn dedup_tables(t: &Traces) -> Vec<(&'static str, Vec<&Main>)> {
    vec![
        mains("LT", &t.lts),
        mains("BRANCH", &t.branches),
        mains("BYTEWISE", &t.bytewises),
        mains("EQ", &t.eqs),
        mains("MUL", &t.muls),
        mains("DVRM", &t.dvrms),
    ]
}

/// ★ Sorted: two builds of one run's traces are equal in all six dedup tables,
/// each one chunk of at least 64 rows. In `HashMap` order (the control) every
/// one of the six differs between the two builds. Either way they hold the
/// same rows.
#[test]
#[ignore = "builds a run's traces twice (the 2^20-row BITWISE table): run on the box, once per order"]
fn every_dedup_table_of_a_run_is_reproducible() {
    let (x, y) = (build_traces(RUN), build_traces(RUN));
    for ((name, x), (_, y)) in dedup_tables(&x).into_iter().zip(dedup_tables(&y)) {
        assert_eq!(x.len(), 1, "{name}: one chunk");
        assert!(x[0].height >= 64, "{name}: {} rows", x[0].height);
        assert_eq!(row_multiset(x[0]), row_multiset(y[0]), "{name}: the rows");
        let equal = x == y;
        assert_eq!(
            equal,
            canonical_rows(),
            "{name} in {}: two builds are {}",
            order(),
            if equal { "equal" } else { "different" }
        );
        println!(
            "CANON TRACES {name}: {} rows, two builds {} in {}",
            x[0].height,
            if equal { "equal" } else { "different" },
            order()
        );
    }
}

/// Blowup 4 (the block's) and no grinding, so no nonce search moves a byte.
fn bytes_options() -> ProofOptions {
    let opts = GoldilocksCubicProofOptions::with_params(4, 128, 0).expect("options");
    assert_eq!(opts.grinding_factor, 0, "no nonce search");
    opts
}

fn proof_bytes(proof: &VmProof) -> Vec<u8> {
    rkyv::to_bytes::<rkyv::rancor::Error>(proof)
        .expect("serialize")
        .to_vec()
}

/// ★ Sorted: two proves of one run, each executing it and building its own
/// traces, give the same bytes. In `HashMap` order (the control) they differ.
/// Either way the first proof verifies.
#[test]
#[ignore = "proves a run twice at blowup 4 (the 2^20-row BITWISE table): run on the box, once per order"]
fn a_run_proves_the_same_bytes_twice() {
    let elf_bytes = asm_elf_bytes(RUN);
    let opts = bytes_options();
    let prove =
        || crate::prove_with_options(&elf_bytes, &opts, &MaxRowsConfig::default()).expect("prove");
    let first = prove();
    let (x, y) = (proof_bytes(&first), proof_bytes(&prove()));
    assert!(
        matches!(
            crate::verify_with_options(&first, &elf_bytes, &opts, None, None),
            Ok(true)
        ),
        "the proof does not verify in {}",
        order()
    );
    let equal = x == y;
    assert_eq!(
        equal,
        canonical_rows(),
        "two proves of {RUN} in {} are {}",
        order(),
        if equal { "equal" } else { "different" }
    );
    println!(
        "CANON BYTES {RUN}: two proves of {} and {} bytes, {} in {}; the first verifies",
        x.len(),
        y.len(),
        if equal { "equal" } else { "different" },
        order()
    );
}
