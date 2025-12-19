// Acknowledgement: Lambdaclass Ethrex Team (https://github.com/lambdaclass/ethrex)
use clap::Parser;
use report::{LinesOfCodeReport, LinesOfCodeReporterOptions, shell_summary};
use spinoff::{Color, Spinner, spinners::Dots};
use std::{collections::HashMap, fs::DirEntry, path::PathBuf};
use tokei::{Config, Language, LanguageType, Languages};

mod report;

const EXCLUDED: &[&str] = &["tooling", "*target*", "*tests*", "*bench*"];

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

            // Skip excluded directories
            if EXCLUDED.contains(&crate_name) {
                return None;
            }

            if let Some(crate_loc) = count_loc(crate_path.clone(), config) {
                Some((
                    crate_name.to_owned(),
                    crate_loc.code,
                ))
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

fn main() {
    let opts = LinesOfCodeReporterOptions::parse();

    let mut spinner = Spinner::new(Dots, "Counting lines of code...", Color::Cyan);

    // Find the root of the repo
    let repo_path = std::env::var("CARGO_MANIFEST_DIR")
        .map(PathBuf::from)
        .map(|path| path.parent().unwrap().parent().unwrap().to_path_buf())
        .unwrap();
    let repo_crates_path = repo_path.join(""); // TODO: change to "crates" when crates directory exists
    let config = Config::default();

    let lambda_vm_loc = count_loc(repo_path, &config).unwrap();
    let crates_loc = count_crates_loc(&repo_crates_path, &config);

    spinner.success("Lines of code calculated!");

    let mut spinner = Spinner::new(Dots, "Generating report...", Color::Cyan);

    let new_report = LinesOfCodeReport {
        lambda_vm: lambda_vm_loc.code,
        crates: crates_loc,
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
