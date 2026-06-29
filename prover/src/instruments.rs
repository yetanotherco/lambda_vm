use std::collections::BTreeMap;
use std::time::Duration;

fn fmt_rows(rows: usize) -> String {
    if rows >= 1_000_000 {
        format!("{:.1}M", rows as f64 / 1_000_000.0)
    } else if rows >= 1_000 {
        format!("{}K", (rows + 500) / 1_000)
    } else {
        format!("{rows}")
    }
}

fn pct(dur: Duration, total: Duration) -> f64 {
    if total > Duration::ZERO {
        dur.as_secs_f64() / total.as_secs_f64() * 100.0
    } else {
        0.0
    }
}

/// Top-level row: % in first column.
fn row_top(label: &str, dur: Duration, total: Duration) {
    eprintln!(
        "  {:<36} {:>7.2}s  {:>5.1}%",
        label,
        dur.as_secs_f64(),
        pct(dur, total),
    );
}

/// Sub-level row: % shifted right into its own column.
fn row_sub(label: &str, dur: Duration, total: Duration) {
    eprintln!(
        "  {:<36} {:>7.2}s         {:>5.1}%",
        label,
        dur.as_secs_f64(),
        pct(dur, total),
    );
}

/// Strip the `[N]` suffix to get the base table name.
fn base_name(name: &str) -> &str {
    name.find('[').map_or(name, |i| &name[..i])
}

struct MergedTable {
    total_dur: Duration,
    total_rows: usize,
    count: usize,
    sub_ops: stark::instruments::TableSubOps,
}

/// Print a unified timing report to stderr.
pub fn print_report(
    execute: Duration,
    trace_build: Duration,
    air_construction: Duration,
    total: Duration,
    heap_profile: &stark::instruments::ProveHeapProfile,
) {
    let mp = stark::instruments::take();

    eprintln!();
    eprintln!("=== PROVER TIMING ===");
    eprintln!("  {:<36} {:>8}  {:>5}", "Phase", "Wall", "%",);
    eprintln!("  {}", "─".repeat(58));

    row_top("Execute", execute, total);
    row_top("Trace build", trace_build, total);
    row_top("AIR construction", air_construction, total);

    if let Some(ref mp) = mp {
        let round1 = mp.main_commits + mp.aux_build + mp.aux_commit;

        row_top("Pre-pass (domains/twiddles)", mp.prepass, total);
        row_top("Round 1", round1, total);
        row_sub("  Main trace commits", mp.main_commits, total);
        row_sub(
            "    Main LDE (fused GPU: LDE+Keccak+Merkle / CPU: LDE only)",
            mp.round1_sub.main_lde,
            total,
        );
        row_sub(
            "    Main commit (Merkle, CPU only)",
            mp.round1_sub.main_merkle,
            total,
        );
        row_sub("  Aux trace build (parallel)", mp.aux_build, total);
        row_sub("  Aux trace commit", mp.aux_commit, total);
        row_sub(
            "    Aux LDE (fused GPU: LDE+Keccak+Merkle / CPU: LDE only)",
            mp.round1_sub.aux_lde,
            total,
        );
        row_sub(
            "    Aux commit (Merkle, CPU only)",
            mp.round1_sub.aux_merkle,
            total,
        );
        row_top("Rounds 2\u{2013}4", mp.rounds_2_4, total);

        // Merge split tables: MEMW[0..4] → MEMW x5
        let mut merged: BTreeMap<String, MergedTable> = BTreeMap::new();
        for (name, rows, dur, sub_ops) in &mp.table_timings {
            let base = base_name(name).to_string();
            let entry = merged.entry(base).or_insert(MergedTable {
                total_dur: Duration::ZERO,
                total_rows: 0,
                count: 0,
                sub_ops: stark::instruments::TableSubOps::default(),
            });
            entry.total_dur += *dur;
            entry.total_rows += rows;
            entry.count += 1;
            entry.sub_ops.constraints += sub_ops.constraints;
            entry.sub_ops.comp_decompose += sub_ops.comp_decompose;
            entry.sub_ops.comp_commit += sub_ops.comp_commit;
            entry.sub_ops.ood += sub_ops.ood;
            entry.sub_ops.deep_comp += sub_ops.deep_comp;
            entry.sub_ops.deep_extend += sub_ops.deep_extend;
            entry.sub_ops.fri_commit += sub_ops.fri_commit;
            entry.sub_ops.queries += sub_ops.queries;
        }

        let mut sorted: Vec<_> = merged.into_iter().collect();
        sorted.sort_by(|a, b| b.1.total_dur.cmp(&a.1.total_dur));

        let threshold = total.as_secs_f64() * 0.02;
        let mut others_dur = Duration::ZERO;
        let mut others_count = 0usize;

        for (name, t) in &sorted {
            if t.total_dur.as_secs_f64() >= threshold {
                let display_name = if t.count > 1 {
                    format!("{name} x{}", t.count)
                } else {
                    name.clone()
                };
                let label = format!("    {:<18} {:>6}", display_name, fmt_rows(t.total_rows),);
                row_sub(&label, t.total_dur, total);
            } else {
                others_dur += t.total_dur;
                others_count += 1;
            }
        }
        if others_count > 0 {
            let label = format!("    ({others_count} others)");
            row_sub(&label, others_dur, total);
        }

        // Sub-operation totals across all tables
        let mut total_constraints = Duration::ZERO;
        let mut total_comp_decompose = Duration::ZERO;
        let mut total_comp_commit = Duration::ZERO;
        let mut total_ood = Duration::ZERO;
        let mut total_deep_comp = Duration::ZERO;
        let mut total_deep_extend = Duration::ZERO;
        let mut total_fri_commit = Duration::ZERO;
        let mut total_queries = Duration::ZERO;
        for (_, t) in &sorted {
            total_constraints += t.sub_ops.constraints;
            total_comp_decompose += t.sub_ops.comp_decompose;
            total_comp_commit += t.sub_ops.comp_commit;
            total_ood += t.sub_ops.ood;
            total_deep_comp += t.sub_ops.deep_comp;
            total_deep_extend += t.sub_ops.deep_extend;
            total_fri_commit += t.sub_ops.fri_commit;
            total_queries += t.sub_ops.queries;
        }

        let sub_ops_sum = total_constraints
            + total_comp_decompose
            + total_comp_commit
            + total_ood
            + total_deep_comp
            + total_deep_extend
            + total_fri_commit
            + total_queries;
        if sub_ops_sum > Duration::ZERO {
            let mut sub_ops: Vec<(&str, Duration)> = vec![
                ("R2  evaluate", total_constraints),
                ("R2  decompose_and_extend_d2", total_comp_decompose),
                ("R2  commit_composition_poly", total_comp_commit),
                ("R3  OOD evaluation", total_ood),
                ("R4  deep_composition_poly_evals", total_deep_comp),
                ("R4  interpolate+evaluate_fft", total_deep_extend),
                ("R4  fri::commit_phase", total_fri_commit),
                ("R4  queries & openings", total_queries),
            ];
            sub_ops.sort_by(|a, b| b.1.cmp(&a.1));
            eprintln!(
                "  {}",
                "    \u{2500}\u{2500} sub-operation totals (all tables) \u{2500}\u{2500}",
            );
            for (label, dur) in &sub_ops {
                row_sub(&format!("    {label}"), *dur, total);
            }
        }

        // Cross-round totals: all FFT work and all Merkle work
        let total_fft = mp.round1_sub.main_lde
            + mp.round1_sub.aux_lde
            + total_comp_decompose
            + total_deep_extend;
        let total_merkle = mp.round1_sub.main_merkle
            + mp.round1_sub.aux_merkle
            + total_comp_commit
            + total_fri_commit;
        eprintln!();
        eprintln!(
            "  {:<36} {:>7.2}s  {:>5.1}%",
            "Total FFT",
            total_fft.as_secs_f64(),
            pct(total_fft, total)
        );
        eprintln!(
            "  {:<36} {:>7.2}s  {:>5.1}%",
            "Total Merkle",
            total_merkle.as_secs_f64(),
            pct(total_merkle, total)
        );
    }

    eprintln!("  {}", "─".repeat(58));
    eprintln!("  {:<36} {:>7.2}s", "TOTAL", total.as_secs_f64());
    eprintln!();

    let mb = |b: usize| b / (1024 * 1024);
    if let Some(before) = heap_profile.before {
        eprintln!("=== HEAP PROFILE (MB) ===");
        eprintln!("  {:<36} {:>8} {:>8}", "Phase", "Heap", "Delta");
        eprintln!("  {}", "─".repeat(56));
        let mut prev = before;
        let mut print_row = |label: &str, val: Option<usize>| {
            if let Some(v) = val {
                let cur = mb(v);
                let delta = mb(v) as isize - mb(prev) as isize;
                eprintln!("  {:<36} {:>7} {:>+8}", label, cur, delta);
                prev = v;
            }
        };
        print_row("After execute", heap_profile.after_execute);
        print_row("After trace build", heap_profile.after_trace_build);
        print_row("After AIR construction", heap_profile.after_air);
        if let Some(ref mp_data) = mp {
            for (label, bytes) in &mp_data.heap_snapshots {
                let cur = mb(*bytes);
                let delta = cur as isize - mb(prev) as isize;
                eprintln!("  {:<36} {:>7} {:>+8}", label, cur, delta);
                prev = *bytes;
            }
        }
        eprintln!("  {}", "─".repeat(56));
        eprintln!();
    }
}
