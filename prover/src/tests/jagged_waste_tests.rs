//! How much of the committed trace is padding — the ceiling on what a jagged PCS
//! could save over the power-of-two stacking the multilinear path commits today.
//!
//! Four cell counts per proof, all in base-field elements:
//! - `real`:    every table's rows before padding, times its main width.
//! - `padded`:  every table rounded up to its power-of-two height.
//! - `stacked`: what `global_layout` commits — `num_polys · 2^n_stack`.
//! - `jagged`:  the real cells packed contiguously, rounded up to whole
//!   `2^n_stack` polynomials of the same size the stack uses.
//! Both again with polynomials 8x smaller (`fine`), which separates the tail
//! of the last polynomial — any stacking can shrink it — from the per-table
//! padding only jagged removes.
//!
//! `LAMBDA_VM_JAGGED_PROGRAMS` picks the private inputs (comma-separated
//! `executor/tests/*.bin` stems), all run against the `ethrex` ELF, and
//! `LAMBDA_VM_JAGGED_EPOCHS` the epoch sizes (comma-separated log2; `0` is the
//! monolithic proof).

use std::collections::BTreeMap;

use executor::elf::Elf;
use executor::vm::execution::Executor;
use multilinear::stacking::StackedLayout;
use stark::multilinear_table::global_layout;

use crate::MaxRowsConfig;
use crate::tables::trace_builder::{DecodeArtifacts, Traces};

type Records = Vec<(&'static str, usize, usize)>;

fn artifact(rel: &str) -> Vec<u8> {
    let path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("workspace root")
        .join(rel);
    std::fs::read(&path).unwrap_or_else(|_| panic!("read {}", path.display()))
}

fn normalize(s: &str) -> String {
    s.chars()
        .filter(|c| c.is_ascii_alphanumeric())
        .map(|c| c.to_ascii_lowercase())
        .collect()
}

fn pct(x: u64, of: u64) -> f64 {
    100.0 * x as f64 / of.max(1) as f64
}

fn take_records() -> Records {
    std::mem::take(&mut *crate::REAL_ROWS.lock().unwrap())
}

#[derive(Default)]
struct Totals {
    proofs: usize,
    real: u64,
    padded: u64,
    stacked: u64,
    jagged: u64,
    /// Same two, with polynomials 8x smaller than today's `n_stack`.
    stacked_fine: u64,
    jagged_fine: u64,
    rows_real: u64,
    rows_padded: u64,
    /// kind -> (instances, width, real rows, padded rows, waste cells)
    kinds: BTreeMap<String, (usize, usize, u64, u64, u64)>,
    unmatched: Vec<String>,
    leftover: Vec<String>,
    /// Every table in AIR order: (kind, width, real rows, padded rows).
    tables: Vec<(String, usize, usize, usize)>,
}

impl Totals {
    fn add(&mut self, other: Totals) {
        self.proofs += other.proofs;
        self.real += other.real;
        self.padded += other.padded;
        self.stacked += other.stacked;
        self.jagged += other.jagged;
        self.stacked_fine += other.stacked_fine;
        self.jagged_fine += other.jagged_fine;
        self.rows_real += other.rows_real;
        self.rows_padded += other.rows_padded;
        for (k, v) in other.kinds {
            let e = self.kinds.entry(k).or_default();
            e.0 += v.0;
            e.1 = v.1;
            e.2 += v.2;
            e.3 += v.3;
            e.4 += v.4;
        }
        self.unmatched.extend(other.unmatched);
        self.leftover.extend(other.leftover);
    }

    fn summary(&self) -> String {
        format!(
            "real {} padded {} ({:.1}% padding) stacked {} ({:.1}% not real) jagged {} ({:.1}% not real) | jagged vs stacked {:.1}% fewer cells | fine: stacked {:.1}% / jagged {:.1}% not real, jagged {:.1}% fewer | rows padding {:.1}%",
            self.real,
            self.padded,
            pct(self.padded - self.real, self.padded),
            self.stacked,
            pct(self.stacked - self.real, self.stacked),
            self.jagged,
            pct(self.jagged - self.real, self.jagged),
            pct(self.stacked - self.jagged, self.stacked),
            pct(self.stacked_fine - self.real, self.stacked_fine),
            pct(self.jagged_fine - self.real, self.jagged_fine),
            pct(self.stacked_fine - self.jagged_fine, self.stacked_fine),
            pct(self.rows_padded - self.rows_real, self.rows_padded),
        )
    }

    fn print_kinds(&self) {
        let waste_total = self.padded - self.real;
        let mut kinds: Vec<_> = self
            .kinds
            .iter()
            .filter(|(k, e)| !k.starts_with("fixed:") || e.4 > 0)
            .collect();
        kinds.sort_by_key(|(_, e)| std::cmp::Reverse(e.4));
        println!(
            "{:<16} {:>5} {:>6} {:>12} {:>12} {:>7} {:>14} {:>7}",
            "kind", "inst", "width", "real_rows", "pad_rows", "pad%", "waste_cells", "%waste"
        );
        for (kind, (inst, width, r, h, waste)) in kinds {
            println!(
                "{kind:<16} {inst:>5} {width:>6} {r:>12} {h:>12} {:>6.1}% {waste:>14} {:>6.1}%",
                pct(h - r, *h),
                pct(*waste, waste_total),
            );
        }
        println!("unmatched tables (treated as fully real): {:?}", self.unmatched);
        println!("records with no table: {:?}", self.leftover);
    }
}

/// Matches every table to the real-row record its trace generator left, by
/// kind and padded height, and counts the cells.
fn analyze(pairs: &[crate::AirTracePair<'_>], records: Records) -> Totals {
    let mut by_kind: BTreeMap<String, Vec<(usize, usize)>> = BTreeMap::new();
    for (kind, real, padded) in records {
        by_kind
            .entry(normalize(kind))
            .or_default()
            .push((padded, real));
    }
    let kinds: Vec<String> = by_kind.keys().cloned().collect();

    let mut t = Totals {
        proofs: 1,
        ..Default::default()
    };
    let mut shapes = Vec::with_capacity(pairs.len());
    for (air, trace, _) in pairs {
        let name = air.name().to_string();
        let height = trace.main_table.height;
        let width = trace.main_table.width;
        // Two AIRs are named by abbreviation, not by their module, and CPU[32]
        // is a CPU chunk, not the CPU32 table.
        let norm = normalize(&name);
        let norm = if let Some(rest) = norm.strip_prefix("memwa") {
            format!("memwaligned{rest}")
        } else if let Some(rest) = norm.strip_prefix("memwr") {
            format!("memwregister{rest}")
        } else if name.starts_with("CPU[") {
            "cpu".to_string()
        } else if name == "unknown" {
            // The epoch bookend's AIR carries no name of its own.
            "localtoglobal".to_string()
        } else {
            norm
        };
        let kind = kinds
            .iter()
            .filter(|k| norm.starts_with(k.as_str()))
            .max_by_key(|k| k.len())
            .cloned();
        let real = match kind.as_ref().and_then(|k| {
            let list = by_kind.get_mut(k).unwrap();
            let at = list.iter().position(|&(p, _)| p == height)?;
            Some(list.swap_remove(at).1)
        }) {
            Some(real) => real,
            None => {
                t.unmatched.push(format!("{name}({height}x{width})"));
                height
            }
        };
        let kind = kind.unwrap_or_else(|| format!("fixed:{}", norm.split("0x").next().unwrap()));
        t.tables.push((kind.clone(), width, real, height));
        let e = t.kinds.entry(kind).or_default();
        e.0 += 1;
        e.1 = width;
        e.2 += real as u64;
        e.3 += height as u64;
        e.4 += ((height - real) * width) as u64;
        t.real += (real * width) as u64;
        t.padded += (height * width) as u64;
        t.rows_real += real as u64;
        t.rows_padded += height as u64;
        shapes.push((width, height.trailing_zeros() as usize));
    }
    t.leftover = by_kind
        .iter()
        .flat_map(|(k, v)| v.iter().map(move |(p, r)| format!("{k}({r}/{p})")))
        .collect();
    // Jagged packs the real cells contiguously into polynomials of the same
    // size the stack uses, so the two differ only in the per-table padding.
    let layout = global_layout(&shapes).expect("layout");
    let n = layout.n_stack();
    t.stacked = (layout.num_polys() as u64) << n;
    t.jagged = t.real.div_ceil(1u64 << n) << n;
    let heights: Vec<usize> = shapes
        .iter()
        .flat_map(|&(w, m)| std::iter::repeat_n(m, w))
        .collect();
    let tallest = heights.iter().copied().max().unwrap_or(0);
    let fine = n.saturating_sub(3).max(tallest);
    let fine_layout = StackedLayout::build(&heights, fine).expect("fine layout");
    t.stacked_fine = (fine_layout.num_polys() as u64) << fine;
    t.jagged_fine = t.real.div_ceil(1u64 << fine) << fine;
    t
}

fn options() -> crate::ProofOptions {
    stark::proof::options::GoldilocksCubicProofOptions::with_params(4, 128, 20)
        .expect("valid options")
}

fn monolithic(elf: &Elf, inputs: &[u8]) -> Totals {
    let logs = Executor::new(elf, inputs.to_vec())
        .and_then(Executor::run)
        .expect("run")
        .logs;
    take_records();
    let mut traces =
        Traces::from_elf_and_logs(elf, &logs, &MaxRowsConfig::default(), inputs).expect("traces");
    drop(logs);
    let records = take_records();
    let table_counts = traces.table_counts();
    let airs = crate::VmAirs::new(
        elf,
        &options(),
        false,
        &traces.page_configs,
        &table_counts,
        None,
        true,
        None,
        None,
        None,
    );
    let pairs = airs.air_trace_pairs(&mut traces);
    analyze(&pairs, records)
}

/// Every epoch's tables exactly as `multilinear_continuation::prove_epoch`
/// assembles them, the bookend included.
fn epochs(elf: &Elf, inputs: &[u8], epoch_size_log2: u32) -> Totals {
    let opts = options();
    take_records();
    let artifacts = DecodeArtifacts::from_elf(elf).expect("decode artifacts");
    // DECODE is built once from the ELF and reused by every epoch.
    let decode: Records = take_records()
        .into_iter()
        .filter(|&(kind, _, _)| kind == "decode")
        .collect();
    let mut total = Totals::default();
    crate::continuation::for_each_epoch(elf, inputs, epoch_size_log2, &artifacts, |prepared, _| {
        let mut traces = prepared.traces;
        let reg_fini = crate::tables::register::fini_from_trace(&traces.register);
        let table_counts = traces.table_counts();
        let airs = crate::continuation::build_epoch_airs(
            elf,
            &opts,
            &[],
            &table_counts,
            &prepared.register_init,
            &reg_fini,
            prepared.is_final,
            None,
        );
        let l2g_air = crate::continuation::l2g_memory_air(&opts, prepared.label);
        let mut l2g_trace =
            crate::tables::local_to_global::generate_local_to_global_trace(&prepared.boundary);
        let mut records = take_records();
        records.extend(decode.iter().copied());
        let mut pairs = airs.air_trace_pairs(&mut traces);
        pairs.push((&l2g_air, &mut l2g_trace, &()));
        let t = analyze(&pairs, records);
        if std::env::var("LAMBDA_VM_CELLS_ELF").is_err() {
            println!("  epoch {:>3}: {}", prepared.index, t.summary());
        }
        total.add(t);
        Ok(())
    })
    .expect("epochs");
    total
}

#[test]
#[ignore]
fn jagged_waste() {
    let programs = std::env::var("LAMBDA_VM_JAGGED_PROGRAMS")
        .unwrap_or_else(|_| "ethrex_10_transfers".to_string());
    let sizes = std::env::var("LAMBDA_VM_JAGGED_EPOCHS").unwrap_or_else(|_| "0".to_string());
    let elf_bytes = artifact("executor/program_artifacts/rust/ethrex.elf");
    let elf = Elf::load(&elf_bytes).expect("elf");
    for input in programs.split(',').filter(|s| !s.is_empty()) {
        let inputs = artifact(&format!("executor/tests/{input}.bin"));
        for size in sizes.split(',').filter(|s| !s.is_empty()) {
            let size: u32 = size.parse().expect("epoch size log2");
            let label = if size == 0 {
                "monolithic".to_string()
            } else {
                format!("epochs 2^{size}")
            };
            println!("\n=== {input} — {label}");
            let t = if size == 0 {
                monolithic(&elf, &inputs)
            } else {
                epochs(&elf, &inputs, size)
            };
            println!("TOTAL ({} proofs): {}", t.proofs, t.summary());
            t.print_kinds();
        }
    }
}

/// Proves the epochs on the multilinear path, for the phase timers
/// `prove_epoch` and `multi_prove` print (`ML_EPOCH_TIMING`, `ML_MULTI_PROVE`).
#[test]
#[ignore]
fn jagged_phase_timing() {
    let input = std::env::var("LAMBDA_VM_JAGGED_PROGRAMS")
        .unwrap_or_else(|_| "ethrex_10_transfers".to_string());
    let size: u32 = std::env::var("LAMBDA_VM_JAGGED_EPOCHS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(22);
    let elf_bytes = artifact("executor/program_artifacts/rust/ethrex.elf");
    let inputs = artifact(&format!("executor/tests/{input}.bin"));
    let opts = stark::proof::options::GoldilocksCubicProofOptions::with_blowup(2).expect("opts");
    let start = std::time::Instant::now();
    let proofs =
        crate::multilinear_continuation::prove_epochs(&elf_bytes, &inputs, size, &opts).expect("prove");
    let prove_s = start.elapsed().as_secs_f64();
    if let Ok(path) = std::env::var("LAMBDA_VM_JAGGED_DUMP") {
        let archived = rkyv::to_bytes::<rkyv::rancor::Error>(&proofs).expect("serialize");
        std::fs::write(&path, &archived).expect("write the proofs");
    }
    let bytes: usize = proofs
        .iter()
        .map(|p| rkyv::to_bytes::<rkyv::rancor::Error>(p).expect("serialize").len())
        .sum();
    crypto::hash_metrics::reset();
    multilinear::jagged::EVAL_MULS.store(0, std::sync::atomic::Ordering::Relaxed);
    let start = std::time::Instant::now();
    let ok = crate::multilinear_continuation::verify_epochs(&elf_bytes, &proofs, &opts)
        .expect("verify");
    let hashes = crypto::hash_metrics::snapshot();
    eprintln!(
        "ML_VERIFY_HASHES jagged={} {hashes:?} jagged_eval_muls={}",
        stark::multilinear_table::jagged_enabled(),
        multilinear::jagged::EVAL_MULS.load(std::sync::atomic::Ordering::Relaxed)
    );
    eprintln!(
        "ML_TOTAL jagged={} epochs={} wall_s={prove_s:.3} verify_s={:.3} verified={ok} proof_bytes={bytes}",
        stark::multilinear_table::jagged_enabled(),
        proofs.len(),
        start.elapsed().as_secs_f64()
    );
    assert!(ok, "the epochs do not verify");
}

/// The streaming prover's commitment groups (`multilinear_retire::plan`),
/// approximated from the monolithic tables: every unsplit table in one group,
/// KECCAK_RND alone, each split kind's chunks `k` to a group, the pages
/// together. Cells each group commits stacked versus jagged.
#[test]
#[ignore]
fn a1_group_waste() {
    let elf_bytes = artifact("executor/program_artifacts/rust/ethrex.elf");
    let elf = Elf::load(&elf_bytes).expect("elf");
    let input = std::env::var("LAMBDA_VM_JAGGED_PROGRAMS")
        .unwrap_or_else(|_| "ethrex_mainnet_25368371".to_string());
    let inputs = artifact(&format!("executor/tests/{input}.bin"));
    let t = monolithic(&elf, &inputs);
    let split = [
        "cpu", "lt", "memw", "memwaligned", "load", "mul", "dvrm", "shift", "branch",
        "memwregister", "eq", "bytewise", "store", "cpu32",
    ];
    for k in [1usize, 4] {
        let mut groups: Vec<Vec<&(String, usize, usize, usize)>> = Vec::new();
        let mut head = Vec::new();
        let mut pages = Vec::new();
        let mut by_kind: BTreeMap<&str, Vec<&(String, usize, usize, usize)>> = BTreeMap::new();
        for table in &t.tables {
            if table.0.starts_with("fixed:page") {
                pages.push(table);
            } else if table.0 == "keccakrnd" {
                groups.push(vec![table]);
            } else if split.contains(&table.0.as_str()) {
                by_kind.entry(table.0.as_str()).or_default().push(table);
            } else {
                head.push(table);
            }
        }
        groups.push(head);
        groups.push(pages);
        for chunks in by_kind.values() {
            for group in chunks.chunks(k) {
                groups.push(group.to_vec());
            }
        }
        let (mut stacked, mut jagged, mut real, mut padded) = (0u64, 0u64, 0u64, 0u64);
        let mut worst: Vec<(f64, String, u64, u64)> = Vec::new();
        for group in &groups {
            let shapes: Vec<(usize, usize)> =
                group.iter().map(|g| (g.1, g.3.trailing_zeros() as usize)).collect();
            let layout = global_layout(&shapes).expect("layout");
            let s_cells = (layout.num_polys() as u64) << layout.n_stack();
            let widths: Vec<usize> = group.iter().map(|g| g.1).collect();
            let rows: Vec<usize> = group
                .iter()
                .flat_map(|g| std::iter::repeat_n(g.2, g.1))
                .collect();
            let j = multilinear::jagged::JaggedLayout::build(&widths, &rows, 25).expect("jagged");
            let j_cells = j.cells() as u64;
            let r: u64 = group.iter().map(|g| (g.1 * g.2) as u64).sum();
            let p: u64 = group.iter().map(|g| (g.1 * g.3) as u64).sum();
            stacked += s_cells;
            jagged += j_cells;
            real += r;
            padded += p;
            worst.push((
                s_cells as f64 / j_cells.max(1) as f64,
                format!("{}x{}", group[0].0, group.len()),
                s_cells,
                j_cells,
            ));
        }
        worst.sort_by(|a, b| b.2.saturating_sub(b.3).cmp(&a.2.saturating_sub(a.3)));
        println!(
            "\n=== {input} A1 k={k}: {} groups | real {real} padded {padded} | stacked {stacked} ({:.1}% not real) | jagged {jagged} ({:.1}% not real) | jagged vs stacked {:.1}% fewer cells",
            groups.len(),
            pct(stacked - real, stacked),
            pct(jagged - real, jagged),
            pct(stacked - jagged, stacked),
        );
        for (_, name, s_cells, j_cells) in worst.iter().take(8) {
            println!("  {name:<20} stacked {s_cells:>12} jagged {j_cells:>12} saves {:>12}", s_cells - j_cells.min(s_cells));
        }
    }
}

/// Which tables carry preprocessed columns in an epoch, and how many cells the
/// verifier evaluates for them.
#[test]
#[ignore]
fn preprocessed_census() {
    let elf_bytes = artifact("executor/program_artifacts/rust/ethrex.elf");
    let elf = Elf::load(&elf_bytes).expect("elf");
    let input = std::env::var("LAMBDA_VM_JAGGED_PROGRAMS")
        .unwrap_or_else(|_| "ethrex_10_transfers".to_string());
    let inputs = artifact(&format!("executor/tests/{input}.bin"));
    let opts = options();
    let artifacts = DecodeArtifacts::from_elf(&elf).expect("decode artifacts");
    let mut first = true;
    crate::continuation::for_each_epoch(&elf, &inputs, 22, &artifacts, |prepared, _| {
        if !first {
            return Ok(());
        }
        first = false;
        let mut traces = prepared.traces;
        let reg_fini = crate::tables::register::fini_from_trace(&traces.register);
        let table_counts = traces.table_counts();
        let airs = crate::continuation::build_epoch_airs(
            &elf, &opts, &[], &table_counts, &prepared.register_init, &reg_fini,
            prepared.is_final, None,
        );
        let pairs = airs.air_trace_pairs(&mut traces);
        let mut total = 0u64;
        for (air, trace, _) in &pairs {
            let p = air.num_precomputed_columns();
            if p > 0 {
                let cells = (p * trace.main_table.height) as u64;
                total += cells;
                println!("{:<16} precomputed {:>3} x {:>9} rows = {:>11} cells", air.name(), p, trace.main_table.height, cells);
            }
        }
        println!("TOTAL preprocessed cells evaluated by the verifier: {total}");
        Ok(())
    })
    .expect("epochs");
}

/// BITWISE's closed form is its columns' extension: at points in the extension
/// field, not just on the cube, it gives what evaluating the 2^20 rows gives.
#[test]
fn bitwise_closed_form_matches_its_columns() {
    use crate::tables::bitwise;
    use math::field::element::FieldElement;
    type X = FieldElement<crate::tables::types::GoldilocksExtension>;
    let columns: Vec<multilinear::mle::Mle<crate::test_utils::F>> = bitwise::preprocessed_columns()
        .into_iter()
        .map(|c| multilinear::mle::Mle::new(c).unwrap())
        .collect();
    let g = |v: u64| FieldElement::<crate::test_utils::F>::from(v);
    for seed in 0..3u64 {
        let point: Vec<X> = (0..20u64)
            .map(|i| {
                let a = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ (i * 0x1234_5678_9ABC);
                X::new([g(a % 1_000_000_007), g(a.rotate_left(17) % 998_244_353), g(i + 3 + seed)])
            })
            .collect();
        let closed = bitwise::precomputed_closed_form(&point);
        assert_eq!(closed.len(), columns.len());
        for (col, column) in columns.iter().enumerate() {
            assert_eq!(closed[col], column.evaluate_in(&point).unwrap(), "column {col}, seed {seed}");
        }
    }
    // And on the cube, row by row.
    for row in [0usize, 1, 255, 256, 65_535, 65_536, 123_457, (1 << 20) - 1] {
        let point: Vec<X> = (0..20).map(|i| X::from(((row >> (19 - i)) & 1) as u64)).collect();
        let closed = bitwise::precomputed_closed_form(&point);
        let expected = bitwise::generate_bitwise_row(row);
        for (col, value) in expected.iter().enumerate() {
            assert_eq!(closed[col], X::from(*value), "row {row} column {col}");
        }
    }
}

/// Evaluating columns together against one `eq` table gives what folding each
/// one on its own gives, zeros skipped and all.
#[test]
fn evaluating_columns_together_matches_one_by_one() {
    use math::field::element::FieldElement;
    use multilinear::mle::Mle;
    type X = FieldElement<crate::tables::types::GoldilocksExtension>;
    let g = |v: u64| FieldElement::<crate::test_utils::F>::from(v);
    let columns: Vec<Mle<crate::test_utils::F>> = (0..6u64)
        .map(|c| {
            Mle::new(
                (0..(1u64 << 12))
                    .map(|i| if (i * 7 + c) % 3 == 0 { g(0) } else { g(i.wrapping_mul(2654435761) ^ c) })
                    .collect(),
            )
            .unwrap()
        })
        .collect();
    let refs: Vec<&Mle<crate::test_utils::F>> = columns.iter().collect();
    let point: Vec<X> = (0..12u64)
        .map(|i| X::new([g(i * 1_000_003 + 5), g(i + 77), g(i * i + 1)]))
        .collect();
    let together = Mle::evaluate_many_in(&refs, &point).unwrap();
    for (column, value) in columns.iter().zip(&together) {
        assert_eq!(*value, column.evaluate_in(&point).unwrap());
    }
}

/// DECODE's streamed evaluation is its columns' extension, padding and all.
#[test]
fn decode_streamed_evaluation_matches_its_columns() {
    use crate::tables::decode;
    use math::field::element::FieldElement;
    type X = FieldElement<crate::tables::types::GoldilocksExtension>;
    let g = |v: u64| FieldElement::<crate::test_utils::F>::from(v);
    for program in ["executor/program_artifacts/rust/ethrex.elf", "executor/program_artifacts/rust/keccak.elf"] {
        let elf = Elf::load(&artifact(program)).expect("elf");
        let instructions = decode::instructions_from_elf(&elf).expect("decode");
        let columns: Vec<multilinear::mle::Mle<crate::test_utils::F>> =
            decode::preprocessed_columns(&instructions)
                .into_iter()
                .map(|c| multilinear::mle::Mle::new(c).unwrap())
                .collect();
        let n = columns[0].num_vars();
        let point: Vec<X> = (0..n as u64)
            .map(|i| X::new([g(i * 7_654_321 + 11), g(i + 5), g(i * i * 3 + 2)]))
            .collect();
        let streamed = decode::evaluate_preprocessed(&instructions, &point);
        for (col, column) in columns.iter().enumerate() {
            assert_eq!(streamed[col], column.evaluate_in(&point).unwrap(), "{program} column {col}");
        }
    }
}

/// The trace cells a guest's execution fills, epoch by epoch — what proving a
/// recursive verifier costs. `LAMBDA_VM_CELLS_ELF` and `LAMBDA_VM_CELLS_INPUT`
/// are absolute paths.
#[test]
#[ignore]
fn guest_trace_cells() {
    let elf_bytes = std::fs::read(std::env::var("LAMBDA_VM_CELLS_ELF").expect("elf path")).expect("elf");
    let inputs = std::fs::read(std::env::var("LAMBDA_VM_CELLS_INPUT").expect("input path")).expect("input");
    let elf = Elf::load(&elf_bytes).expect("elf");
    let t = epochs(&elf, &inputs, 22);
    println!("CELLS proofs={} real={} padded={}", t.proofs, t.real, t.padded);
    let mut kinds: Vec<_> = t.kinds.iter().collect();
    kinds.sort_by_key(|(_, e)| std::cmp::Reverse(e.3 * e.1 as u64));
    for (kind, (inst, width, real, padded, _)) in kinds.iter().take(12) {
        println!("  {kind:<16} inst {inst:>5} width {width:>5} real_cells {:>14} padded_cells {:>14}", real * *width as u64, padded * *width as u64);
    }
}
