// Acknowledgement: Lambdaclass Ethrex Team (https://github.com/lambdaclass/ethrex)
use clap::Parser;
use report::{LinesOfCodeReport, LinesOfCodeReporterOptions, shell_summary};
use spinoff::{Color, Spinner, spinners::Dots};
use std::{collections::HashMap, fs::DirEntry, path::PathBuf};
use tokei::{Config, Language, LanguageType, Languages};

mod report;

const EXCLUDED: &[&str] = &[
    "tooling",
    "bench_vs",
    "*target*",
    "*tests*",
    "*test_utils*",
    "*bench*",
    "*benches*",
    "*examples*",
    // Non-production code: fuzz harnesses and the RISC-V guest programs
    // executed/proven by the zkVM (test fixtures, not the zkVM itself).
    "*fuzz*",
    "*programs*",
    "*program_artifacts*",
    // Formal-verification harnesses (z3/QF-BV gates, …): not Rust, not part of
    // the zkVM itself — counted as their own standalone report section.
    "formal_verification",
];

/// Directories counted separately (not as crates).
const CRATE_SKIPPED: &[&str] = &["tooling", "bin", "formal_verification"];

fn count_crates_loc(crates_path: &PathBuf, config: &Config) -> Vec<(String, usize)> {
    let top_level_crate_dirs = std::fs::read_dir(crates_path)
        .unwrap()
        .filter_map(|e| e.ok())
        .collect::<Vec<DirEntry>>();

    let mut crates_loc: Vec<(String, usize)> = top_level_crate_dirs
        .into_iter()
        .filter_map(|crate_dir_entry| {
            let crate_path = crate_dir_entry.path();
            let crate_name = crate_path.file_name().unwrap().to_str().unwrap();

            // Skip excluded and separately-counted directories
            if EXCLUDED.contains(&crate_name) || CRATE_SKIPPED.contains(&crate_name) {
                return None;
            }

            if let Some(crate_loc) = count_loc(crate_path.clone(), config) {
                Some((crate_name.to_owned(), crate_loc.code))
            } else {
                None
            }
        })
        .collect();

    crates_loc.sort_by_key(|(_crate_name, loc)| *loc);
    crates_loc.reverse();
    crates_loc
}

fn count_loc(path: PathBuf, config: &Config) -> Option<Language> {
    let mut languages = Languages::new();
    languages.get_statistics(&[path], EXCLUDED, config);
    languages.get(&LanguageType::Rust).cloned()
}

fn count_tools_loc(bin_path: &PathBuf, config: &Config) -> Vec<(String, usize)> {
    if !bin_path.exists() {
        return Vec::new();
    }

    let tool_dirs = std::fs::read_dir(bin_path)
        .unwrap()
        .filter_map(|e| e.ok())
        .collect::<Vec<DirEntry>>();

    let mut tools_loc: Vec<(String, usize)> = tool_dirs
        .into_iter()
        .filter_map(|tool_dir_entry| {
            let tool_path = tool_dir_entry.path();
            let tool_name = tool_path.file_name().unwrap().to_str().unwrap().to_owned();

            // Only count directories (crates)
            if !tool_path.is_dir() {
                return None;
            }

            let mut languages = Languages::new();
            // Use a subset of exclusions for tools
            let tool_excluded: &[&str] = &["*target*", "*tests*", "*bench*", "*benches*"];
            languages.get_statistics(&[tool_path], tool_excluded, config);
            languages
                .get(&LanguageType::Rust)
                .map(|rust_loc| (tool_name, rust_loc.code))
        })
        .collect();

    tools_loc.sort_by_key(|(_tool_name, loc)| *loc);
    tools_loc.reverse();
    tools_loc
}

fn count_formal_verification_loc(fv_path: &PathBuf, config: &Config) -> Vec<(String, usize)> {
    if !fv_path.exists() {
        return Vec::new();
    }

    let gate_dirs = std::fs::read_dir(fv_path)
        .unwrap()
        .filter_map(|e| e.ok())
        .collect::<Vec<DirEntry>>();

    let mut fv_loc: Vec<(String, usize)> = gate_dirs
        .into_iter()
        .filter_map(|gate_dir_entry| {
            let gate_path = gate_dir_entry.path();

            // Only count directories (one per verified chip/gate)
            if !gate_path.is_dir() {
                return None;
            }

            let gate_name = gate_path.file_name().unwrap().to_str().unwrap().to_owned();

            // The harnesses are not Rust (python/z3 today, possibly Lean or SMT
            // later), so sum code lines across every language tokei recognizes.
            let mut languages = Languages::new();
            languages.get_statistics(&[gate_path], &[], config);
            let gate_loc: usize = languages.values().map(|language| language.code).sum();
            (gate_loc > 0).then_some((gate_name, gate_loc))
        })
        .collect();

    fv_loc.sort_by_key(|(_gate_name, loc)| *loc);
    fv_loc.reverse();
    fv_loc
}

fn main() {
    let opts = LinesOfCodeReporterOptions::parse();

    let mut spinner = Spinner::new(Dots, "Counting lines of code...", Color::Cyan);

    // Find the root of the repo
    let repo_path = std::env::var("CARGO_MANIFEST_DIR")
        .map(PathBuf::from)
        .map(|path| path.parent().unwrap().parent().unwrap().to_path_buf())
        .unwrap();
    let repo_crates_path = repo_path.join(""); // TODO: change to "crates" when crates directory exists
    let repo_bin_path = repo_path.join("bin");
    let repo_formal_verification_path = repo_path.join("formal_verification");
    let config = Config::default();

    let lambda_vm_loc = count_loc(repo_path.clone(), &config).unwrap();
    let crates_loc = count_crates_loc(&repo_crates_path, &config);
    let tools_loc = count_tools_loc(&repo_bin_path, &config);
    let formal_verification_loc =
        count_formal_verification_loc(&repo_formal_verification_path, &config);

    spinner.success("Lines of code calculated!");

    let mut spinner = Spinner::new(Dots, "Generating report...", Color::Cyan);

    let new_report = LinesOfCodeReport {
        lambda_vm: lambda_vm_loc.code,
        crates: crates_loc,
        tools: tools_loc,
        formal_verification: formal_verification_loc,
    };

    if opts.detailed {
        let mut current_detailed_loc_report = HashMap::new();
        for report in lambda_vm_loc.reports {
            let file_path = report.name;
            current_detailed_loc_report
                .entry(file_path.as_os_str().to_str().unwrap().to_owned())
                .and_modify(|e: &mut usize| *e += report.stats.code)
                .or_insert_with(|| report.stats.code);
        }

        std::fs::write(
            "current_detailed_loc_report.json",
            serde_json::to_string(&current_detailed_loc_report).unwrap(),
        )
        .expect("current_detailed_loc_report.json could not be written");
    } else if opts.compare_detailed {
        let current_detailed_loc_report: HashMap<String, usize> =
            std::fs::read_to_string("current_detailed_loc_report.json")
                .map(|s| serde_json::from_str(&s).unwrap())
                .expect("current_detailed_loc_report.json could not be read");

        let previous_detailed_loc_report: HashMap<String, usize> =
            std::fs::read_to_string("previous_detailed_loc_report.json")
                .map(|s| serde_json::from_str(&s).unwrap())
                .unwrap_or(current_detailed_loc_report.clone());

        std::fs::write(
            "detailed_loc_report.txt",
            report::pr_message(previous_detailed_loc_report, current_detailed_loc_report),
        )
        .unwrap();
    } else if opts.summary {
        spinner.success("Report generated!");
        println!("{}", shell_summary(new_report));
    } else {
        std::fs::write(
            "loc_report.json",
            serde_json::to_string(&new_report).unwrap(),
        )
        .expect("loc_report.json could not be written");

        let old_report: LinesOfCodeReport = std::fs::read_to_string("loc_report.json.old")
            .map(|s| serde_json::from_str(&s).unwrap())
            .unwrap_or(new_report.clone());

        std::fs::write(
            "loc_report_slack.txt",
            report::slack_message(old_report.clone(), new_report.clone()),
        )
        .unwrap();
        std::fs::write(
            "loc_report_github.txt",
            report::github_step_summary(old_report, new_report),
        )
        .unwrap();

        spinner.success("Report generated!");
    }
}
