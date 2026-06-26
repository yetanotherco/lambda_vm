// Acknowledgement: Lambdaclass Ethrex Team (https://github.com/lambdaclass/ethrex)
use clap::Parser;
use colored::Colorize;
use prettytable::{Table, row};
use serde::{Deserialize, Serialize};
use hashbrown::HashMap;

#[derive(Parser)]
pub struct LinesOfCodeReporterOptions {
    #[arg(short, long, value_name = "SUMMARY", default_value = "false")]
    pub summary: bool,
    #[arg(short, long, value_name = "DETAILED", default_value = "false")]
    pub detailed: bool,
    #[arg(short, long, value_name = "PR_SUMMARY", default_value = "false")]
    pub compare_detailed: bool,
}

#[derive(Default, Serialize, Deserialize, Clone)]
pub struct LinesOfCodeReport {
    pub lambda_vm: usize,
    pub crates: Vec<(String, usize)>,
    #[serde(default)]
    pub tools: Vec<(String, usize)>,
}

pub fn pr_message(
    old_report: HashMap<String, usize>,
    new_report: HashMap<String, usize>,
) -> String {
    let sorted_file_paths = {
        let mut keys: Vec<_> = new_report.keys().collect();
        keys.sort();
        keys
    };

    let mut table = Table::new();

    table.add_row(row!["File", "Lines", "Diff"]);

    let mut total_lines_changed: i64 = 0;
    let mut total_lines_added: i64 = 0;
    let mut total_lines_removed: i64 = 0;

    for file_path in sorted_file_paths {
        let current_loc = *new_report.get(file_path).unwrap() as i64;
        let previous_loc = *old_report.get(file_path).unwrap_or(&0) as i64;
        let loc_diff = current_loc - previous_loc;

        if loc_diff == 0 {
            continue;
        }

        if loc_diff > 0 {
            total_lines_added += loc_diff;
        } else {
            total_lines_removed += loc_diff.abs();
        }

        total_lines_changed += loc_diff.abs();

        // remove "lambda_vm/" and everything before it
        const LAMBDA_VM_PREFIX: &str = "lambda_vm/";
        let file_path_printable = if let Some(idx) = file_path.find(LAMBDA_VM_PREFIX) {
            &file_path[idx + LAMBDA_VM_PREFIX.len()..]
        } else {
            file_path
        };

        table.add_row(row![
            file_path_printable,
            current_loc,
            match current_loc.cmp(&previous_loc) {
                std::cmp::Ordering::Greater => format!("+{loc_diff}"),
                std::cmp::Ordering::Less => format!("{loc_diff}"),
                std::cmp::Ordering::Equal => "-".to_owned(),
            }
        ]);
    }

    if total_lines_changed == 0 {
        return "".to_string();
    }

    let mut pr_message = String::new();

    pr_message.push_str("<h2>Lines of code report</h2>\n");
    pr_message.push('\n');

    pr_message.push_str(&pr_message_summary(
        total_lines_added,
        total_lines_removed,
        total_lines_changed,
    ));

    pr_message.push('\n');
    pr_message.push_str("<details>\n");
    pr_message.push_str("<summary>Detailed view</summary>\n");
    pr_message.push('\n');
    pr_message.push_str("```\n");
    pr_message.push_str(&format!("{table}\n"));
    pr_message.push_str("```\n");
    pr_message.push_str("</details>\n");

    pr_message
}

fn pr_message_summary(
    total_lines_added: i64,
    total_lines_removed: i64,
    total_lines_changed: i64,
) -> String {
    let mut pr_message = String::new();

    pr_message.push_str(&format!(
        "Total lines added: `{}`\n",
        match total_lines_added.cmp(&0) {
            std::cmp::Ordering::Greater => format!("{total_lines_added}"),
            std::cmp::Ordering::Less =>
                unreachable!("total_lines_added should never be less than 0"),
            std::cmp::Ordering::Equal => format!("{total_lines_added}"),
        }
    ));
    pr_message.push_str(&format!(
        "Total lines removed: `{}`\n",
        match total_lines_removed.cmp(&0) {
            std::cmp::Ordering::Greater | std::cmp::Ordering::Equal =>
                format!("{total_lines_removed}"),
            std::cmp::Ordering::Less =>
                unreachable!("total_lines_removed should never be less than 0"),
        }
    ));
    pr_message.push_str(&format!(
        "Total lines changed: `{}`\n",
        match total_lines_changed.cmp(&0) {
            std::cmp::Ordering::Greater | std::cmp::Ordering::Equal =>
                format!("{total_lines_changed}"),
            std::cmp::Ordering::Less =>
                unreachable!("total_lines_changed should never be less than 0"),
        }
    ));

    pr_message
}

pub fn slack_message(old_report: LinesOfCodeReport, new_report: LinesOfCodeReport) -> String {
    let diff_total = new_report.lambda_vm.abs_diff(old_report.lambda_vm);

    let crates_mrkdwn = new_report
        .crates
        .iter()
        .fold(String::new(), |acc, (crate_name, loc)| {
            let old_loc = old_report
                .crates
                .iter()
                .find(|(old_crate_name, _)| old_crate_name == crate_name)
                .map(|(_, old_loc)| old_loc)
                .unwrap_or(&0);

            let loc_diff = loc.abs_diff(*old_loc);
            format!(
                "{}*{}*: {} {}\\n",
                acc,
                crate_name,
                loc,
                match loc.cmp(old_loc) {
                    std::cmp::Ordering::Greater => format!("(+{loc_diff})"),
                    std::cmp::Ordering::Less => format!("(-{loc_diff})"),
                    std::cmp::Ordering::Equal => "".to_string(),
                }
            )
        });

    let tools_mrkdwn = new_report
        .tools
        .iter()
        .fold(String::new(), |acc, (tool_name, loc)| {
            let old_loc = old_report
                .tools
                .iter()
                .find(|(old_tool_name, _)| old_tool_name == tool_name)
                .map(|(_, old_loc)| old_loc)
                .unwrap_or(&0);

            let loc_diff = loc.abs_diff(*old_loc);
            format!(
                "{}*{}*: {} {}\\n",
                acc,
                tool_name,
                loc,
                match loc.cmp(old_loc) {
                    std::cmp::Ordering::Greater => format!("(+{loc_diff})"),
                    std::cmp::Ordering::Less => format!("(-{loc_diff})"),
                    std::cmp::Ordering::Equal => "".to_string(),
                }
            )
        });

    let tools_block = if !new_report.tools.is_empty() {
        format!(
            r#",
        {{
            "type": "header",
            "text": {{
                "type": "plain_text",
                "text": "Tools"
            }}
        }},
        {{
            "type": "section",
            "text": {{
                "type": "mrkdwn",
                "text": "{}"
            }}
        }}"#,
            tools_mrkdwn
        )
    } else {
        String::new()
    };

    format!(
        r#"{{
    "blocks": [
        {{
            "type": "header",
            "text": {{
                "type": "plain_text",
                "text": "Daily Lines of Code Report"
            }}
        }},
        {{
            "type": "divider"
        }},
        {{
            "type": "header",
            "text": {{
                "type": "plain_text",
                "text": "Summary"
            }}
        }},
        {{
            "type": "section",
            "text": {{
                "type": "mrkdwn",
                "text": "*lambda_vm (total):* {} {}"
            }}
        }},
        {{
            "type": "header",
            "text": {{
                "type": "plain_text",
                "text": "Crates"
            }}
        }},
        {{
            "type": "section",
            "text": {{
                "type": "mrkdwn",
                "text": "{}"
            }}
        }}{}
    ]
}}"#,
        new_report.lambda_vm,
        match new_report.lambda_vm.cmp(&old_report.lambda_vm) {
            std::cmp::Ordering::Greater => format!("(+{diff_total})"),
            std::cmp::Ordering::Less => format!("(-{diff_total})"),
            std::cmp::Ordering::Equal => "".to_string(),
        },
        crates_mrkdwn,
        tools_block
    )
}

pub fn github_step_summary(old_report: LinesOfCodeReport, new_report: LinesOfCodeReport) -> String {
    let diff_total = new_report.lambda_vm.abs_diff(old_report.lambda_vm);

    let crates_github = new_report
        .crates
        .iter()
        .fold(String::new(), |acc, (crate_name, loc)| {
            let old_loc = old_report
                .crates
                .iter()
                .find(|(old_crate_name, _)| old_crate_name == crate_name)
                .map(|(_, old_loc)| old_loc)
                .unwrap_or(&0);

            let loc_diff = loc.abs_diff(*old_loc);
            format!(
                "{}{}: {} {}\n",
                acc,
                crate_name,
                loc,
                match loc.cmp(old_loc) {
                    std::cmp::Ordering::Greater => format!("(+{loc_diff})"),
                    std::cmp::Ordering::Less => format!("(-{loc_diff})"),
                    std::cmp::Ordering::Equal => "".to_string(),
                }
            )
        });

    let tools_github = if !new_report.tools.is_empty() {
        let tools_list = new_report
            .tools
            .iter()
            .fold(String::new(), |acc, (tool_name, loc)| {
                let old_loc = old_report
                    .tools
                    .iter()
                    .find(|(old_tool_name, _)| old_tool_name == tool_name)
                    .map(|(_, old_loc)| old_loc)
                    .unwrap_or(&0);

                let loc_diff = loc.abs_diff(*old_loc);
                format!(
                    "{}{}: {} {}\n",
                    acc,
                    tool_name,
                    loc,
                    match loc.cmp(old_loc) {
                        std::cmp::Ordering::Greater => format!("(+{loc_diff})"),
                        std::cmp::Ordering::Less => format!("(-{loc_diff})"),
                        std::cmp::Ordering::Equal => "".to_string(),
                    }
                )
            });
        format!("\nlambda_vm tools loc\n=================\n{}", tools_list)
    } else {
        String::new()
    };

    format!(
        r#"```
lambda_vm loc summary
====================
lambda_vm (total): {} {}

lambda_vm crates loc
=================
{}{}
```"#,
        new_report.lambda_vm,
        if new_report.lambda_vm > old_report.lambda_vm {
            format!("(+{diff_total})")
        } else {
            format!("(-{diff_total})")
        },
        crates_github,
        tools_github
    )
}

pub fn shell_summary(new_report: LinesOfCodeReport) -> String {
    let tools_line = if !new_report.tools.is_empty() {
        format!(
            "\n{} {}",
            "tools:".bold(),
            new_report
                .tools
                .iter()
                .map(|(name, loc)| format!("{}: {}", name, loc))
                .collect::<Vec<_>>()
                .join(", ")
        )
    } else {
        String::new()
    };

    format!(
        "{}\n{}\n{} {}\n{} {}{}",
        "Lines of Code".bold(),
        "=============".bold(),
        "lambda_vm (total):".bold(),
        new_report.lambda_vm,
        "crates:".bold(),
        new_report
            .crates
            .iter()
            .map(|(name, loc)| format!("{}: {}", name, loc))
            .collect::<Vec<_>>()
            .join(", "),
        tools_line
    )
}
