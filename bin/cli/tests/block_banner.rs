//! The block commands' `ZF FORMAT` banner states the format they run under:
//! `--blowup 2` prints `whir_blowup=2` on the one banner line (and Q 229 on the
//! schedule line), the default and `--blowup 4` name no blowup (Q 114), and the
//! format's environment knob for the blowup is refused before any banner. The
//! banner is printed before the files are read, so a missing file is enough.

use std::process::Command;

fn run(args: &[&str], extra: &[(&str, &str)]) -> (bool, String, String) {
    let mut cmd = Command::new(env!("CARGO_BIN_EXE_cli"));
    cmd.args(args);
    for (name, _) in std::env::vars() {
        if name.starts_with("LAMBDA_VM_ZF_") || name == "LAMBDA_VM_WHIR_GROUP" {
            cmd.env_remove(name);
        }
    }
    for (name, value) in extra {
        cmd.env(name, value);
    }
    let out = cmd.output().expect("the CLI runs");
    (
        out.status.success(),
        String::from_utf8_lossy(&out.stdout).into_owned(),
        String::from_utf8_lossy(&out.stderr).into_owned(),
    )
}

fn lines<'a>(stdout: &'a str, prefix: &str) -> Vec<&'a str> {
    stdout.lines().filter(|l| l.starts_with(prefix)).collect()
}

#[test]
fn the_block_commands_banner_states_the_blowup_they_run_under() {
    let dir = tempfile::tempdir().expect("a scratch directory");
    let missing = dir.path().join("missing");
    let missing = missing.to_str().expect("a UTF-8 path");
    let proof = dir.path().join("block.proof");
    let proof = proof.to_str().expect("a UTF-8 path");
    let commands: [(&str, Vec<&str>); 2] = [
        ("prove-block", vec!["prove-block", missing, "-o", proof]),
        ("verify-block", vec!["verify-block", missing, missing]),
    ];
    for (name, base) in &commands {
        for (flag, blowup_word, q) in [
            (None, None, " q=114 "),
            (Some("4"), None, " q=114 "),
            (Some("2"), Some(" whir_blowup=2"), " q=229 "),
        ] {
            let mut args = base.clone();
            if let Some(f) = flag {
                args.extend(["--blowup", f]);
            }
            let (ok, stdout, stderr) = run(&args, &[]);
            let tag = format!("{name} --blowup {flag:?}");
            assert!(!ok, "{tag}: a missing file fails the command");
            let banners = lines(&stdout, "ZF FORMAT:");
            assert_eq!(banners.len(), 1, "{tag}: one banner\n{stdout}\n{stderr}");
            match blowup_word {
                Some(word) => assert!(banners[0].ends_with(word), "{tag}: {}", banners[0]),
                None => assert!(!banners[0].contains("whir_blowup"), "{tag}: {}", banners[0]),
            }
            let schedules = lines(&stdout, "ZF WHIR SCHEDULES:");
            assert_eq!(schedules.len(), 1, "{tag}: one schedule line\n{stdout}");
            assert!(schedules[0].contains(q), "{tag}: {}", schedules[0]);
        }
        let (ok, stdout, stderr) = run(base, &[("LAMBDA_VM_ZF_WHIR_BLOWUP", "2")]);
        assert!(!ok, "{name}: the environment knob is refused");
        assert!(
            lines(&stdout, "ZF FORMAT:").is_empty(),
            "{name}: no banner before the refusal\n{stdout}"
        );
        assert!(
            stderr.contains("give the CLI --blowup instead"),
            "{name}: the refusal names the flag\n{stderr}"
        );
    }
}
