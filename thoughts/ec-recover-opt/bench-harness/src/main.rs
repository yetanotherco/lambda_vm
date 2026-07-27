//! Per-table cell breakdown for the phase-H EC-share measurement.
//!
//! `prover::count_elements_by_table` is a library function with no CLI
//! subcommand, so this thin binary exposes it. Prints every table's committed
//! base cells (`main + 3·aux`, one extension element being 3 base elements) and
//! the EC share, sorted by cost.
//!
//! Usage: ec-bench-harness <elf> [private_input_file]

use std::process::ExitCode;

/// Tables that constitute the elliptic-curve accelerator.
const EC_TABLES: &[&str] = &["ECSM", "ECDAS", "ECSM2", "ECDAS2", "EC_T0"];

fn main() -> ExitCode {
    let mut args = std::env::args().skip(1);
    let Some(elf_path) = args.next() else {
        eprintln!("usage: ec-bench-harness <elf> [private_input_file]");
        return ExitCode::FAILURE;
    };
    let elf = match std::fs::read(&elf_path) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("cannot read {elf_path}: {e}");
            return ExitCode::FAILURE;
        }
    };
    let input = match args.next() {
        Some(p) => match std::fs::read(&p) {
            Ok(b) => b,
            Err(e) => {
                eprintln!("cannot read {p}: {e}");
                return ExitCode::FAILURE;
            }
        },
        None => Vec::new(),
    };

    let rows = match prover::count_elements_by_table(&elf, &input) {
        Ok(r) => r,
        Err(e) => {
            eprintln!("count_elements_by_table failed: {e:?}");
            return ExitCode::FAILURE;
        }
    };

    // committed base cells = main + 3 * aux (aux is counted in EF columns)
    let mut table: Vec<(&str, u64)> = rows
        .iter()
        .map(|(name, main, aux)| (*name, main + 3 * aux))
        .collect();
    let total: u64 = table.iter().map(|(_, c)| *c).sum();
    let ec: u64 = table
        .iter()
        .filter(|(n, _)| EC_TABLES.contains(n))
        .map(|(_, c)| *c)
        .sum();

    table.sort_by_key(|(_, c)| std::cmp::Reverse(*c));
    println!("{:<24} {:>16} {:>8}", "table", "base cells", "share");
    for (name, cells) in &table {
        if *cells == 0 {
            continue;
        }
        let mark = if EC_TABLES.contains(name) { " <- EC" } else { "" };
        println!(
            "{:<24} {:>16} {:>7.2}%{}",
            name,
            cells,
            100.0 * *cells as f64 / total as f64,
            mark
        );
    }
    println!();
    println!("total base cells : {total}");
    println!("EC base cells    : {ec}");
    println!(
        "EC SHARE         : {:.2}%",
        100.0 * ec as f64 / total as f64
    );
    println!();
    println!("NOTE: only meaningful on a REAL ethrex block. On small guests the");
    println!("fixed-size tables (BITWISE is a fixed 2^16 rows, PAGE is fixed per");
    println!("page) dominate and the share is an artefact. See BENCH.md section 4.1.");
    ExitCode::SUCCESS
}
