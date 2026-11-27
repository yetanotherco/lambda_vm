//! Process runner for the block levels over a configured TCP topology.
//!
//! `cluster prove` publishes each run's proof file at OUTPUT first and checks
//! it right after, including when a range join uses early leaf artifacts. Neither
//! OUTPUT nor its marker carries trust. A consumer reads the
//! `(OwnedBlockStatement, file)` pair, compares the statement with the one it
//! expects (program, input, outputs), and verifies the file itself:
//! `verify_cluster_block_levels_file` for a range topology's file (it also
//! reads a one-leaf file), `verify_block_levels_file` for a replica file.
//!
//! When the check refuses run k's file, the runner adds the line
//! `run k: <reason>` to `OUTPUT.refused`, proves the block locally, checks
//! that proof, and replaces OUTPUT with it once. The entry is written before
//! the local proof starts. If writing it, the local proof or its check fails,
//! the process exits with an error and OUTPUT still holds the refused file
//! (with the marker entry, unless writing the marker was what failed).
//!
//! What the marker does not say:
//! - No marker, or no entry for a run, does not mean the file passed. The
//!   check runs after publication; a kill, panic or crash before it ends
//!   leaves the unchecked file at OUTPUT and writes no entry.
//! - An entry can describe a file that is no longer at OUTPUT: it stays after
//!   the local replacement and after later runs overwrite OUTPUT.
//! - Entries are keyed by run number only, and run numbers restart at 1 with
//!   each invocation. An accepted run k of a later invocation clears an
//!   earlier invocation's entry for run k; entries above the later
//!   invocation's `--runs` are never cleared.
//!
//! With `--runs`, the check and any recovery finish before the next run
//! starts. Overlapping checks with later blocks is future work because host
//! derivation for the check would compete with the next block for CPU time.

use std::{
    ffi::{OsStr, OsString},
    fs::{self, OpenOptions},
    io::{ErrorKind, Write},
    num::NonZeroUsize,
    os::unix::process::CommandExt,
    path::{Path, PathBuf},
    process::{Command, ExitCode},
    time::Instant,
};

use lambda_vm_prover::{
    block_whir::{
        B3Rounds, BlockFormat, BlockOptions,
        cluster_tcp::{Topology, local_fallbacks, serve_attempts},
    },
    lfm::{
        cluster_levels::{
            UncheckedClusterFile, VerifiedClusterFile, check_shipment,
            prove_cluster_block_levels_unchecked, prove_range_levels_reference,
            prove_range_levels_reference_unchecked, prove_replica_levels_unchecked,
        },
        levels::pin_family,
        proof::block_base_options_under,
        wlfm::WlfmFamily,
    },
};
use multilinear::whir_chain::GroupFold;

type RunError = Box<dyn std::error::Error>;

const USAGE: &str = "usage: cluster [--family poseidon|blake3] worker TOPOLOGY ID [--attempts N] | \
                     cluster [--family poseidon|blake3] prove TOPOLOGY ELF INPUT|- OUTPUT [--runs N] | \
                     cluster --family blake3 reference WINDOWS|- ELF INPUT|- OUTPUT";

fn main() -> ExitCode {
    let args: Vec<_> = std::env::args_os().skip(1).collect();
    match run(&args) {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("cluster: {e}");
            ExitCode::FAILURE
        }
    }
}

fn repetitions(args: &[OsString], flag: &str) -> Result<NonZeroUsize, RunError> {
    match args {
        [] => Ok(NonZeroUsize::MIN),
        [option, count] if option == flag => count
            .to_str()
            .and_then(|count| count.parse().ok())
            .ok_or_else(|| format!("{flag} must be a positive integer").into()),
        _ => Err(USAGE.into()),
    }
}

fn pinned_format(family: WlfmFamily) -> Result<BlockFormat, RunError> {
    Ok(pin_family(
        BlockFormat::pinned_with(GroupFold::Indexed),
        family,
        B3Rounds::Fri,
    )?)
}

fn run(args: &[OsString]) -> Result<(), RunError> {
    let (family, command) = match args {
        [flag, family, command @ ..] if flag == "--family" => {
            let family = match family.to_str() {
                Some("poseidon" | "p1") => WlfmFamily::P1,
                Some("blake3" | "b3") => WlfmFamily::B3,
                _ => return Err("--family must be poseidon or blake3".into()),
            };
            (family, command)
        }
        _ => (WlfmFamily::P1, args),
    };
    match command {
        [mode, topology, id, rest @ ..] if mode == "worker" => {
            let attempts = repetitions(rest, "--attempts")?;
            let topology = Topology::read(Path::new(topology))?;
            let id = id
                .to_str()
                .ok_or("worker index must be a number")?
                .parse()?;
            let cards = topology.worker_cards(id)?;
            if cfg!(feature = "cuda") && std::env::var("LAMBDA_VM_CARDS").as_deref() != Ok(&cards) {
                // CUDA and Rayon read their environment at initialization.
                return Err(Command::new(std::env::current_exe()?)
                    .args(args)
                    .env("LAMBDA_VM_CARDS", cards)
                    .exec()
                    .into());
            }
            let format = pinned_format(family)?;
            let opts = block_base_options_under(&format.zf);
            serve_attempts(&topology, id, &opts, &format, attempts)?;
        }
        [mode, topology, elf, input, output, rest @ ..] if mode == "prove" => {
            let runs = repetitions(rest, "--runs")?;
            let options = BlockOptions::production();
            let topology = Topology::read(Path::new(topology))?;
            let elf = fs::read(elf)?;
            let input = if input == OsStr::new("-") {
                Vec::new()
            } else {
                fs::read(input)?
            };
            prove_runs(
                &topology,
                &elf,
                &input,
                &options,
                family,
                runs,
                Path::new(output),
            )?;
        }
        [mode, cuts, elf, input, output] if mode == "reference" => {
            let cuts = cuts.to_str().ok_or("window boundaries must be text")?;
            let cuts = if cuts == "-" {
                Vec::new()
            } else {
                cuts.split(',')
                    .map(str::parse)
                    .collect::<Result<Vec<usize>, _>>()?
            };
            let elf = fs::read(elf)?;
            let input = if input == OsStr::new("-") {
                Vec::new()
            } else {
                fs::read(input)?
            };
            let verified = prove_range_levels_reference(
                &elf,
                &input,
                &BlockOptions::production(),
                family,
                &cuts,
            )?;
            let clock = verified.stage_clock();
            let file = rkyv::to_bytes::<rkyv::rancor::Error>(&verified.into_parts())?;
            clock.mark("outer-encode-done");
            publish(Path::new(output), &file)?;
            clock.mark("PUBLISHED");
        }
        _ => return Err(USAGE.into()),
    }
    Ok(())
}

fn prove_runs(
    topology: &Topology,
    elf: &[u8],
    input: &[u8],
    options: &BlockOptions,
    family: WlfmFamily,
    runs: NonZeroUsize,
    output: &Path,
) -> Result<(), RunError> {
    let marker = refused_marker(output);
    let cuts = topology.range_cuts();
    for number in 1..=runs.get() {
        let start = Instant::now();
        let fallbacks = local_fallbacks();
        let shipment = if cuts.is_some() {
            prove_cluster_block_levels_unchecked(elf, input, options, family, topology)?
        } else {
            prove_replica_levels_unchecked(elf, input, options, family, Some(topology))?
        };
        let clock = shipment.stage_clock();
        let file = rkyv::to_bytes::<rkyv::rancor::Error>(shipment.contents())?;
        clock.mark("outer-encode-done");
        publish(output, &file)?;
        eprintln!(
            "{}",
            published_line(
                number,
                local_fallbacks() != fallbacks,
                start.elapsed().as_secs_f64()
            )
        );
        clock.mark("PUBLISHED");

        clock.mark("file-check-start");
        let check_start = Instant::now();
        let checked = check_published(number, output, shipment);
        let check_wall = check_start.elapsed().as_secs_f64();
        clock.mark("file-check-end");
        match checked {
            Ok(_) => {
                eprintln!("CLUSTER FILE CHECK {number} ACCEPTED {check_wall:.6}");
                update_refused_marker(&marker, number, None)?;
            }
            Err(reason) => {
                let reason = one_line(&reason);
                eprintln!("CLUSTER FILE CHECK {number} REFUSED {check_wall:.6} {reason}");
                replace_refused(number, output, &marker, &reason, || {
                    let local = match cuts {
                        Some(cuts) => prove_range_levels_reference_unchecked(
                            elf, input, options, family, cuts,
                        ),
                        None => prove_replica_levels_unchecked(elf, input, options, family, None),
                    }?;
                    let checked = check_replacement(number, output, local)?;
                    rkyv::to_bytes::<rkyv::rancor::Error>(&checked.into_parts())
                        .map_err(|e| e.to_string())
                })?;
                eprintln!(
                    "CLUSTER PROVE RUN {number} REPUBLISHED LOCAL WALL {:.6}",
                    start.elapsed().as_secs_f64()
                );
            }
        }
    }
    Ok(())
}

/// The wall line printed once a run's file is published. A run whose TCP
/// attempt fell back to a local proof says so in place of `PUBLISHED`, so a
/// timing read that counts `PUBLISHED WALL` lines never counts a local proof
/// as a cluster run.
fn published_line(number: usize, fell_back: bool, wall: f64) -> String {
    if fell_back {
        format!("CLUSTER PROVE RUN {number} LOCAL FALLBACK WALL {wall:.6}")
    } else {
        format!("CLUSTER PROVE RUN {number} PUBLISHED WALL {wall:.6}")
    }
}

fn check_published(
    _number: usize,
    _output: &Path,
    shipment: UncheckedClusterFile<'_>,
) -> Result<VerifiedClusterFile, String> {
    #[cfg(all(test, feature = "test-security"))]
    tests::tcp::observe_publication(_number, _output)?;
    check_shipment(shipment)
}

fn check_replacement(
    _number: usize,
    _output: &Path,
    local: UncheckedClusterFile<'_>,
) -> Result<VerifiedClusterFile, String> {
    #[cfg(all(test, feature = "test-security"))]
    tests::tcp::observe_replacement(_number, _output)?;
    check_shipment(local)
}

/// After run `number`'s published file was refused: record the refusal in
/// the marker, then build the checked local replacement, then publish it, in
/// that order. On an error before the publication OUTPUT still holds the
/// refused file.
fn replace_refused<B: AsRef<[u8]>>(
    number: usize,
    output: &Path,
    marker: &Path,
    reason: &str,
    replacement: impl FnOnce() -> Result<B, String>,
) -> Result<(), RunError> {
    let refused = |what: String| {
        format!(
            "run {number}: the published file {} was refused ({reason}) and is still in place; {what}",
            output.display()
        )
    };
    update_refused_marker(marker, number, Some(reason))
        .map_err(|e| refused(format!("writing {} failed: {e}", marker.display())))?;
    let bytes = replacement().map_err(|e| {
        refused(format!(
            "its local replacement failed: {e}; {} names the run",
            marker.display()
        ))
    })?;
    publish(output, bytes.as_ref()).map_err(|e| {
        format!("run {number}: publishing the local replacement after a refusal failed: {e}")
    })?;
    Ok(())
}

fn refused_marker(output: &Path) -> PathBuf {
    let mut name = output.as_os_str().to_os_string();
    name.push(".refused");
    name.into()
}

/// A refusal reason as one marker line, so a reason cannot add an entry for
/// another run.
fn one_line(reason: &str) -> String {
    reason.replace(['\r', '\n'], " ")
}

fn update_refused_marker(
    marker: &Path,
    number: usize,
    reason: Option<&str>,
) -> std::io::Result<()> {
    let previous = match fs::read_to_string(marker) {
        Ok(text) => text,
        Err(error) if error.kind() == ErrorKind::NotFound => String::new(),
        Err(error) => return Err(error),
    };
    let prefix = format!("run {number}: ");
    let mut entries = previous
        .lines()
        .filter(|line| !line.starts_with(&prefix))
        .collect::<Vec<_>>();
    let entry;
    if let Some(reason) = reason {
        entry = format!("{prefix}{}", one_line(reason));
        entries.push(&entry);
    } else if entries.len() == previous.lines().count() {
        return Ok(());
    }
    if entries.is_empty() {
        fs::remove_file(marker)?;
        sync_directory(marker)
    } else {
        let mut text = entries.join("\n");
        text.push('\n');
        publish(marker, text.as_bytes())
    }
}

/// How many temporary names `publish` tries before giving up.
const TEMPORARY_NAMES: u32 = 1000;

/// Replace `output` atomically and durably. The bytes go to a new temporary
/// file beside it (so on the same filesystem), which is synced and renamed
/// over `output`; the directory is synced after the rename. Readers opening
/// `output` see the old file or the complete new one. A temporary left by an
/// earlier process with the same pid is skipped, never reused or removed.
fn publish(output: &Path, bytes: &[u8]) -> std::io::Result<()> {
    let (temporary, file) = create_temporary(output)?;
    let result = write_synced(file, bytes).and_then(|()| fs::rename(&temporary, output));
    if result.is_err() {
        let _ = fs::remove_file(temporary);
    }
    result?;
    sync_directory(output)
}

fn create_temporary(output: &Path) -> std::io::Result<(PathBuf, fs::File)> {
    let name = output.file_name().unwrap_or_default();
    let pid = std::process::id();
    let mut index = 0;
    loop {
        let mut temporary = name.to_os_string();
        temporary.push(format!(".{pid}.{index}.tmp"));
        let temporary = output.with_file_name(temporary);
        match OpenOptions::new()
            .create_new(true)
            .write(true)
            .open(&temporary)
        {
            Ok(file) => return Ok((temporary, file)),
            Err(error)
                if error.kind() == ErrorKind::AlreadyExists && index + 1 < TEMPORARY_NAMES =>
            {
                index += 1;
            }
            Err(error) => return Err(error),
        }
    }
}

fn write_synced(mut file: fs::File, bytes: &[u8]) -> std::io::Result<()> {
    file.write_all(bytes)?;
    file.sync_all()
}

/// Sync the directory holding `path`, so a rename or removal there survives
/// a crash.
fn sync_directory(path: &Path) -> std::io::Result<()> {
    let directory = match path.parent() {
        Some(parent) if !parent.as_os_str().is_empty() => parent,
        _ => Path::new("."),
    };
    fs::File::open(directory)?.sync_all()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A fresh directory for one test, named by the test and the pid.
    fn scratch(test: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("cluster-{test}-{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir(&dir).unwrap();
        dir
    }

    #[test]
    fn accepting_a_run_clears_only_its_refusal() {
        let dir = scratch("marker");
        let marker = refused_marker(&dir.join("proof"));
        update_refused_marker(&marker, 2, Some("first reason")).unwrap();
        update_refused_marker(&marker, 20, Some("second reason")).unwrap();
        update_refused_marker(&marker, 1, None).unwrap();
        assert_eq!(
            fs::read_to_string(&marker).unwrap(),
            "run 2: first reason\nrun 20: second reason\n"
        );
        update_refused_marker(&marker, 2, None).unwrap();
        assert_eq!(
            fs::read_to_string(&marker).unwrap(),
            "run 20: second reason\n"
        );
        update_refused_marker(&marker, 20, None).unwrap();
        assert!(!marker.exists());
        fs::remove_dir(dir).unwrap();
    }

    /// ★ A reason holding a line break stays one entry: it cannot leave an
    /// entry for another run after its own run is accepted.
    ///
    /// Mutation: the reason written to the marker as given.
    #[test]
    fn a_reason_cannot_add_an_entry_for_another_run() {
        let dir = scratch("reason");
        let marker = refused_marker(&dir.join("proof"));
        update_refused_marker(&marker, 2, Some("bad\nrun 7: forged\r\nend")).unwrap();
        assert_eq!(
            fs::read_to_string(&marker).unwrap(),
            "run 2: bad run 7: forged  end\n"
        );
        update_refused_marker(&marker, 2, None).unwrap();
        assert!(!marker.exists(), "an entry outlived its run's acceptance");
        fs::remove_dir(dir).unwrap();
    }

    /// ★ A run that fell back to a local proof is named so on its wall
    /// line, and that line does not hold a cluster run's `PUBLISHED WALL`.
    ///
    /// Mutation: the fallback ignored and the line is the cluster run's.
    #[test]
    fn a_run_that_fell_back_is_logged_as_a_fallback() {
        assert_eq!(
            published_line(2, false, 1.5),
            "CLUSTER PROVE RUN 2 PUBLISHED WALL 1.500000"
        );
        let fell_back = published_line(2, true, 1.5);
        assert_eq!(
            fell_back,
            "CLUSTER PROVE RUN 2 LOCAL FALLBACK WALL 1.500000"
        );
        assert!(!fell_back.contains("PUBLISHED WALL"));
    }

    /// ★ After a refusal the marker names the run before the local
    /// replacement starts, while OUTPUT still holds the refused file; the
    /// replacement is then published over it.
    ///
    /// Mutation: the marker written after the replacement is published.
    #[test]
    fn the_refusal_is_recorded_before_the_replacement_starts() {
        let dir = scratch("order");
        let output = dir.join("proof");
        let marker = refused_marker(&output);
        publish(&output, b"refused").unwrap();
        let mut seen = None;
        replace_refused(3, &output, &marker, "bad file", || {
            seen = Some((fs::read_to_string(&marker).ok(), fs::read(&output).unwrap()));
            Ok(b"local".to_vec())
        })
        .unwrap();
        assert_eq!(
            seen,
            Some((Some("run 3: bad file\n".into()), b"refused".to_vec()))
        );
        assert_eq!(fs::read(&output).unwrap(), b"local");
        assert_eq!(fs::read_to_string(&marker).unwrap(), "run 3: bad file\n");
        fs::remove_dir_all(dir).unwrap();
    }

    /// ★ A local replacement that fails ends in an error that says the
    /// refused file is still published; OUTPUT keeps it and the marker
    /// names the run.
    ///
    /// Mutation: a failed replacement ignored and the refusal reported as
    /// handled.
    #[test]
    fn a_failed_replacement_leaves_the_refused_file_and_its_entry() {
        let dir = scratch("failed");
        let output = dir.join("proof");
        let marker = refused_marker(&output);
        publish(&output, b"refused").unwrap();
        let error = replace_refused(4, &output, &marker, "bad file", || {
            Err::<Vec<u8>, _>("the replacement does not verify".into())
        })
        .unwrap_err()
        .to_string();
        assert!(
            error.contains("was refused") && error.contains("does not verify"),
            "{error}"
        );
        assert_eq!(fs::read(&output).unwrap(), b"refused");
        assert_eq!(fs::read_to_string(&marker).unwrap(), "run 4: bad file\n");
        fs::remove_dir_all(dir).unwrap();
    }

    /// ★ When the marker cannot be written, the runner stops before any
    /// replacement and its error says the published file was refused.
    ///
    /// Mutation: the marker write's error ignored.
    #[test]
    fn a_marker_that_cannot_be_written_stops_before_the_replacement() {
        let dir = scratch("unwritable");
        let output = dir.join("proof");
        let marker = refused_marker(&output);
        publish(&output, b"refused").unwrap();
        fs::create_dir(&marker).unwrap();
        let mut replaced = false;
        let error = replace_refused(5, &output, &marker, "bad file", || {
            replaced = true;
            Ok(b"local".to_vec())
        })
        .unwrap_err()
        .to_string();
        assert!(!replaced, "the replacement ran without a marker");
        assert!(
            error.contains("was refused") && error.contains("writing"),
            "{error}"
        );
        assert_eq!(fs::read(&output).unwrap(), b"refused");
        fs::remove_dir_all(dir).unwrap();
    }

    /// ★ A temporary left by an earlier process with the same pid does not
    /// block publication, and is left alone.
    ///
    /// Mutation: one temporary name per pid.
    #[test]
    fn a_stale_temporary_does_not_block_publication() {
        let dir = scratch("stale");
        let output = dir.join("proof");
        let stale = dir.join(format!("proof.{}.0.tmp", std::process::id()));
        fs::write(&stale, b"stale").unwrap();
        publish(&output, b"new").unwrap();
        assert_eq!(fs::read(&output).unwrap(), b"new");
        assert_eq!(fs::read(&stale).unwrap(), b"stale");
        let left = fs::read_dir(&dir).unwrap().count();
        assert_eq!(left, 2, "publication left a temporary behind");
        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn repeat_counts_default_to_one_and_accept_positive_values() {
        for flag in ["--attempts", "--runs"] {
            assert_eq!(repetitions(&[], flag).unwrap().get(), 1);
            assert_eq!(
                repetitions(&[flag.into(), "2".into()], flag).unwrap().get(),
                2
            );
        }
    }

    #[test]
    fn repeat_counts_reject_invalid_arguments() {
        for flag in ["--attempts", "--runs"] {
            for count in ["0", "-1", "two", "999999999999999999999999999999999999"] {
                let error = repetitions(&[flag.into(), count.into()], flag).unwrap_err();
                assert!(error.to_string().contains("positive integer"), "{error}");
            }
            for args in [
                vec![flag.into()],
                vec!["--unknown".into(), "2".into()],
                vec![flag.into(), "2".into(), flag.into(), "3".into()],
            ] {
                assert_eq!(repetitions(&args, flag).unwrap_err().to_string(), USAGE);
            }
        }

        for args in [
            ["worker", "missing-topology", "0", "--runs", "2"].as_slice(),
            [
                "prove",
                "missing-topology",
                "elf",
                "-",
                "out",
                "--attempts",
                "2",
            ]
            .as_slice(),
        ] {
            let args: Vec<_> = args.iter().map(OsString::from).collect();
            assert_eq!(run(&args).unwrap_err().to_string(), USAGE);
        }
    }

    #[cfg(feature = "test-security")]
    pub(super) mod tcp {
        use super::*;
        use lambda_vm_prover::{
            block_whir::OwnedBlockStatement, lfm::cluster_levels::verify_cluster_block_levels_file,
        };
        use std::{cell::RefCell, io::Read, net::TcpListener, os::unix::fs::MetadataExt};

        /// The forced reason holds a line break and a fake entry, so every
        /// forced refusal also shows that the marker keeps one line a run.
        const FORCED_REASON: &str = "forced shipment refusal\nrun 9: forged";
        const FORCED_ENTRY: &str = "forced shipment refusal run 9: forged";

        #[derive(Clone, Copy)]
        enum Refusal {
            None,
            /// The check of this run's published file refuses.
            Shipment(usize),
            /// This run's published file and its local replacement are
            /// both refused.
            Replacement(usize),
        }

        #[derive(Default)]
        struct PublicationProbe {
            refuse: Option<usize>,
            refuse_replacement: Option<usize>,
            /// Run number, bytes and inode at OUTPUT when the check starts.
            published: Vec<(usize, Vec<u8>, u64)>,
            /// Run number, marker text and inode at OUTPUT when the local
            /// replacement's check starts.
            replacements: Vec<(usize, Option<String>, u64)>,
        }

        thread_local! {
            static PROBE: RefCell<PublicationProbe> = RefCell::default();
        }

        fn test_name(name: &str) -> String {
            let (_, path) = module_path!().split_once("::").unwrap();
            format!("{path}::{name}")
        }

        pub(crate) fn observe_publication(number: usize, output: &Path) -> Result<(), String> {
            let mut file = fs::File::open(output).unwrap();
            let inode = file.metadata().unwrap().ino();
            let mut bytes = Vec::new();
            file.read_to_end(&mut bytes).unwrap();
            PROBE.with_borrow_mut(|probe| {
                probe.published.push((number, bytes, inode));
                if probe.refuse == Some(number) {
                    return Err(FORCED_REASON.into());
                }
                Ok(())
            })
        }

        pub(crate) fn observe_replacement(number: usize, output: &Path) -> Result<(), String> {
            let marker = fs::read_to_string(refused_marker(output)).ok();
            let inode = fs::metadata(output).unwrap().ino();
            PROBE.with_borrow_mut(|probe| {
                probe.replacements.push((number, marker, inode));
                if probe.refuse_replacement == Some(number) {
                    return Err("forced replacement refusal".into());
                }
                Ok(())
            })
        }

        #[test]
        #[ignore = "GPU tier: cluster worker subprocess entry"]
        fn worker_process() {
            let path = std::env::var_os("LAMBDA_VM_TEST_REPEAT_TOPOLOGY").unwrap();
            run(&[
                "--family".into(),
                "blake3".into(),
                "worker".into(),
                path,
                "0".into(),
                "--attempts".into(),
                "2".into(),
            ])
            .unwrap();
        }

        fn compare_runs(cuts: &[usize], refusal: Refusal) {
            for var in [
                "LAMBDA_VM_FIXED_TRACE_HASH",
                "LAMBDA_VM_DETERMINISTIC_GRIND",
            ] {
                assert_eq!(std::env::var(var).as_deref(), Ok("1"), "set {var}=1");
            }
            let elf = lambda_vm_prover::test_utils::asm_elf_bytes("test_commit_split");
            let mut options = BlockOptions::production();
            if lambda_vm_prover::security_posture::TEST_POSTURE {
                options.max_rows = lambda_vm_prover::MaxRowsConfig::small();
            }
            if !cuts.is_empty() {
                options.window_log2 = Some(2);
                options.max_rows.cpu = 8;
            }
            let family = WlfmFamily::B3;
            // The clock test compares its two checked runs. The CUDA tier
            // also compares with an independently generated reference.
            let expected = (!lambda_vm_prover::security_posture::TEST_POSTURE).then(|| {
                let reference =
                    prove_range_levels_reference(&elf, &[], &options, family, cuts).unwrap();
                rkyv::to_bytes::<rkyv::rancor::Error>(&reference.into_parts()).unwrap()
            });

            let reserve = || TcpListener::bind("127.0.0.1:0").unwrap();
            let coordinator = reserve();
            let worker = reserve();
            let address = worker.local_addr().unwrap();
            let open_ms = if lambda_vm_prover::security_posture::TEST_POSTURE {
                240000
            } else {
                90000
            };
            let mut text = format!(
                "coordinator {}\nworker {address} 0 {:064x}\ndeadlines-ms 15000 60000 {open_ms}\n",
                coordinator.local_addr().unwrap(),
                1,
            );
            if let [cut] = cuts {
                text.push_str(&format!("windows {cut}\n"));
            }
            let topology = Topology::parse(&text).unwrap();
            let dir = std::env::temp_dir().join(format!(
                "cluster-repeat-{}-{}",
                std::process::id(),
                address.port()
            ));
            fs::create_dir(&dir).unwrap();
            let path = dir.join("topology");
            let output = dir.join("proof");
            let marker = refused_marker(&output);
            update_refused_marker(&marker, 1, Some("prior refusal")).unwrap();
            fs::write(&path, text).unwrap();
            drop(worker);
            let mut worker = Command::new(std::env::current_exe().unwrap());
            worker
                .args([
                    "--exact",
                    &test_name("worker_process"),
                    "--ignored",
                    "--nocapture",
                    "--test-threads=1",
                ])
                .env("LAMBDA_VM_TEST_REPEAT_TOPOLOGY", &path)
                .env("LAMBDA_VM_CARDS", "0")
                .env("RAYON_NUM_THREADS", "2");
            let worker_log = dir.join("worker.log");
            if lambda_vm_prover::security_posture::TEST_POSTURE {
                // Keep the two processes' writes to stderr from interleaving
                // inside a stamp. Forward the worker log after it exits.
                worker.stderr(fs::File::create(&worker_log).unwrap());
            }
            let mut child = worker.spawn().unwrap();
            let runs = repetitions(&["--runs".into(), "2".into()], "--runs").unwrap();
            PROBE.with_borrow_mut(|probe| match refusal {
                Refusal::None => {}
                Refusal::Shipment(number) => probe.refuse = Some(number),
                Refusal::Replacement(number) => {
                    probe.refuse = Some(number);
                    probe.refuse_replacement = Some(number);
                }
            });
            let result = prove_runs(&topology, &elf, &[], &options, family, runs, &output);
            let probe = PROBE.with_borrow_mut(std::mem::take);
            if result.is_err() {
                let _ = child.kill();
            }
            let status = child.wait().unwrap();
            if lambda_vm_prover::security_posture::TEST_POSTURE {
                eprint!("{}", fs::read_to_string(worker_log).unwrap());
            }

            if let Refusal::Replacement(number) = refusal {
                let error = result.expect_err("a refused replacement must end in an error");
                let error = error.to_string();
                assert!(
                    error.contains("was refused") && error.contains("forced replacement refusal"),
                    "{error}"
                );
                assert_eq!(probe.published.len(), number, "no run after the failure");
                let (_, refused, refused_inode) = probe.published.last().unwrap();
                assert_eq!(probe.replacements.len(), 1);
                assert_eq!(fs::read(&output).unwrap(), *refused, "OUTPUT was replaced");
                assert_eq!(fs::metadata(&output).unwrap().ino(), *refused_inode);
                assert_eq!(
                    fs::read_to_string(&marker).unwrap(),
                    format!("run {number}: {FORCED_ENTRY}\n"),
                );
                fs::remove_dir_all(dir).unwrap();
                return;
            }

            result.unwrap();
            assert!(status.success(), "worker exited with {status}");
            assert_eq!(probe.published.len(), 2);
            let expected = expected.as_deref().unwrap_or(&probe.published[0].1);
            for (i, (number, bytes, _)) in probe.published.iter().enumerate() {
                assert_eq!(*number, i + 1);
                assert!(
                    bytes.as_slice() == expected,
                    "published run {number} differs from the checked reference"
                );
            }
            let bytes = fs::read(&output).unwrap();
            assert!(bytes.as_slice() == expected);
            let final_inode = fs::metadata(&output).unwrap().ino();
            let published_inode = probe.published.last().unwrap().2;
            if let Refusal::Shipment(number) = refusal {
                let entry = format!("run {number}: {FORCED_ENTRY}\n");
                // The refusal was on disk, and the refused file still at
                // OUTPUT, when the replacement's check started.
                assert_eq!(
                    probe.replacements,
                    [(number, Some(entry.clone()), published_inode)]
                );
                assert_ne!(final_inode, published_inode, "expected local replacement");
                assert_eq!(fs::read_to_string(&marker).unwrap(), entry);
                let mut aligned = rkyv::util::AlignedVec::<16>::with_capacity(bytes.len());
                aligned.extend_from_slice(&bytes);
                let (statement, file) = rkyv::from_bytes::<
                    (OwnedBlockStatement, Vec<u8>),
                    rkyv::rancor::Error,
                >(&aligned)
                .unwrap();
                verify_cluster_block_levels_file(&elf, statement.view(), family, &file).unwrap();
            } else {
                assert!(probe.replacements.is_empty());
                assert_eq!(final_inode, published_inode);
                assert!(!marker.exists());
            }
            fs::remove_dir_all(dir).unwrap();
        }

        #[test]
        #[ignore = "GPU tier: two TCP replica proofs in the same processes"]
        fn replica_runs_match_local_reference() {
            compare_runs(&[], Refusal::None);
        }

        #[test]
        #[ignore = "GPU tier: two TCP range proofs in the same processes"]
        fn range_runs_match_local_reference() {
            compare_runs(&[2], Refusal::None);
        }

        #[test]
        #[cfg(not(feature = "cuda"))]
        #[ignore = "runs the full TCP range proofs and reference on the host"]
        fn both_range_roles_log_their_stages_in_order() {
            use lambda_vm_prover::block_whir::cluster_tcp::EARLY_JOIN_ENV;
            use std::collections::BTreeMap;

            assert!(
                std::hint::black_box(lambda_vm_prover::security_posture::TEST_POSTURE),
                "run this test in the prover lib-test binary with test-security"
            );
            let output = Command::new(std::env::current_exe().unwrap())
                .args([
                    "--ignored",
                    "--exact",
                    &test_name("a_refused_range_file_is_replaced_with_a_checked_local_proof"),
                    "--nocapture",
                ])
                .env("LAMBDA_VM_FIXED_TRACE_HASH", "1")
                .env("LAMBDA_VM_DETERMINISTIC_GRIND", "1")
                .env("LAMBDA_VM_CLUSTER_EARLY_BIND", "1")
                .env("LAMBDA_VM_CLUSTER_PREPARED_ROOTS_BESIDE", "1")
                .env("LAMBDA_VM_CLUSTER_LEAF_AT_BIND", "1")
                .env(EARLY_JOIN_ENV, "1")
                .env("LAMBDA_VM_CLUSTER_COMPACT_COUNTS", "1")
                .env("RAYON_NUM_THREADS", "4")
                .output()
                .unwrap();
            let stderr = String::from_utf8(output.stderr).unwrap();
            assert!(
                output.status.success(),
                "{}\n{}\n{stderr}",
                output.status,
                String::from_utf8_lossy(&output.stdout)
            );

            let mut lines = stderr.lines();
            for prefix in [
                "CLUSTER PROVE RUN 1 PUBLISHED WALL ",
                "CLUSTER FILE CHECK 1 ACCEPTED ",
                "CLUSTER PROVE RUN 2 PUBLISHED WALL ",
                "CLUSTER FILE CHECK 2 REFUSED ",
                "CLUSTER PROVE RUN 2 REPUBLISHED LOCAL WALL ",
            ] {
                assert!(
                    lines.any(|line| line.starts_with(prefix)),
                    "missing or out of order: {prefix}\n{stderr}"
                );
            }

            let mut attempts = BTreeMap::<_, BTreeMap<_, BTreeMap<_, _>>>::new();
            for line in stderr.lines() {
                let Some(line) = line.strip_prefix("CLUSTER STAGE ") else {
                    continue;
                };
                let fields: Vec<_> = line.split_whitespace().collect();
                let [role, attempt, stage, t, unix, extra @ ..] = fields.as_slice() else {
                    panic!("malformed stage: {line}");
                };
                assert!(matches!(*role, "C" | "W0"), "{line}");
                assert_eq!(attempt.len(), 32, "{line}");
                assert!(attempt.bytes().all(|b| b.is_ascii_hexdigit()), "{line}");
                let unix = unix
                    .strip_prefix("unix_ms=")
                    .unwrap()
                    .parse::<u64>()
                    .unwrap_or_else(|error| panic!("invalid wall clock ({error}): {line}"));
                assert!(unix > 0, "{line}");
                let t = t.strip_prefix("t=").unwrap().parse::<f64>().unwrap();
                if stage.starts_with("base-opening-group-") {
                    let [seconds] = extra else {
                        panic!("missing group seconds: {line}")
                    };
                    assert!(
                        seconds
                            .strip_prefix("seconds=")
                            .unwrap()
                            .parse::<f64>()
                            .unwrap()
                            >= 0.0
                    );
                } else {
                    assert!(extra.is_empty(), "{line}");
                }
                let role_stages = attempts
                    .entry(*attempt)
                    .or_default()
                    .entry(*role)
                    .or_default();
                assert!(
                    role_stages.insert(*stage, t).is_none(),
                    "duplicate stage: {line}"
                );
            }

            let tcp: Vec<_> = attempts
                .values()
                .filter(|roles| roles["C"].contains_key("start-sent-W0"))
                .collect();
            assert_eq!(tcp.len(), 2, "{stderr}");
            let order = |stages: &BTreeMap<&str, f64>, names: &[&str]| {
                for (before, after) in names.iter().zip(&names[1..]) {
                    let t = |name: &str| {
                        *stages
                            .get(name)
                            .unwrap_or_else(|| panic!("missing {name}: {stages:?}"))
                    };
                    assert!(
                        t(before) <= t(after),
                        "{before} must precede {after}: {stages:?}"
                    );
                }
            };
            let leaf = [
                "leaf-plan-done",
                "leaf-emit-done",
                "leaf-compile-done",
                "leaf-validate-done",
                "leaf-prebuild-done",
                "leaf-prepare-done",
                "leaf-build-done",
                "leaf-fill-done",
                "leaf-prove-start",
                "leaf-prove-end",
            ];
            for roles in tcp {
                let c = &roles["C"];
                let w = &roles["W0"];
                order(
                    c,
                    &[
                        "attempt-start",
                        "input-checked",
                        "wait-worker-connect-W0",
                        "worker-connected-W0",
                        "start-sent-W0",
                        "input-sent-W0",
                        "execute-start",
                        "execute-prefix-done",
                        "execute-range-done",
                        "range-build-done",
                        "wait-range-readers",
                        "range-readers-done",
                        "assign-encode-done",
                        "assign-sent-W0",
                        "prepared-roots-done",
                        "wait-receipts",
                        "receipts-received",
                        "receipts-gathered",
                        "seal-done",
                        "bind-encode-done",
                        "bind-sent-W0",
                        "base-openings-start",
                        "base-openings-end",
                        "leaf-arena-done",
                    ],
                );
                order(
                    c,
                    &[
                        "input-sent-W0",
                        "stream-closed-received-W0",
                        "range-ready-received-W0",
                        "range-counts-received-W0",
                        "range-readers-done",
                    ],
                );
                order(
                    c,
                    &[
                        "bind-sent-W0",
                        "leaf-plan-done",
                        "leaf-artifacts-local",
                        "leaf-artifacts-ready",
                    ],
                );
                order(
                    c,
                    &["assign-sent-W0", "final-commits-done", "wait-receipts"],
                );
                order(
                    c,
                    &[
                        "leaf-artifacts-received-W0",
                        "leaf-received-W0",
                        "leaf-relay-done",
                    ],
                );
                order(
                    c,
                    &[
                        "leaf-prove-end",
                        "wait-leaf-relay",
                        "leaf-relay-done",
                        "finish-sent-W0",
                        "wait-finish-W0",
                        "finish-received-W0",
                        "finish-done",
                        "join-fill-done",
                    ],
                );
                order(
                    c,
                    &[
                        "leaf-artifacts-ready",
                        "join-compile-start",
                        "join-compile-end",
                        "join-build-start",
                        "join-build-end",
                        "join-fill-done",
                        "join-prove-start",
                        "join-prove-end",
                        "inner-encode-done",
                        "outer-encode-done",
                        "PUBLISHED",
                        "file-check-start",
                        "file-check-end",
                    ],
                );
                order(
                    w,
                    &[
                        "attempt-start",
                        "wait-input",
                        "input-received",
                        "input-checked",
                        "elf-loaded",
                        "execute-start",
                        "execute-range-done",
                        "range-build-done",
                        "range-ready-sent",
                        "range-counts-sent",
                        "wait-assign",
                        "assign-received",
                        "final-commits-done",
                        "receipts-sent",
                        "wait-bind",
                        "bind-received",
                        "seal-checked",
                        "base-openings-start",
                        "base-openings-end",
                        "leaf-arena-done",
                    ],
                );
                order(
                    w,
                    &[
                        "execute-range-done",
                        "stream-closed-sent",
                        "range-build-done",
                    ],
                );
                order(
                    w,
                    &[
                        "leaf-build-done",
                        "leaf-artifacts-sent",
                        "leaf-prove-start",
                        "leaf-prove-end",
                        "leaf-sent",
                        "opened-all-sent",
                        "wait-finish",
                        "finish-received",
                        "finish-sent",
                    ],
                );
                for stages in [c, w] {
                    order(stages, &leaf);
                    order(stages, &["leaf-arena-done", "leaf-prepare-done"]);
                    order(stages, &["base-openings-end", "leaf-prepare-done"]);
                    order(
                        stages,
                        &[
                            "execute-start",
                            "wait-index-base",
                            "index-base-received",
                            "streamed-commits-done",
                            "range-build-done",
                        ],
                    );
                    let groups: Vec<_> = stages
                        .keys()
                        .filter(|name| name.starts_with("base-opening-group-"))
                        .collect();
                    assert!(!groups.is_empty());
                    for group in groups {
                        order(stages, &["base-openings-start", group, "base-openings-end"]);
                    }
                }
                order(w, &["seal-checked", "leaf-plan-done"]);
                order(c, &["leaf-artifacts-received-W0", "leaf-artifacts-ready"]);
            }
        }

        #[test]
        #[ignore = "GPU tier: refusal after publication and checked local re-proof"]
        fn a_refused_range_file_is_replaced_with_a_checked_local_proof() {
            compare_runs(&[2], Refusal::Shipment(2));
        }

        #[test]
        #[ignore = "GPU tier: replica refusal after publication and checked local re-proof"]
        fn a_refused_replica_file_is_replaced_with_a_checked_local_proof() {
            compare_runs(&[], Refusal::Shipment(2));
        }

        #[test]
        #[ignore = "GPU tier: a refused file whose local replacement is refused too"]
        fn a_refused_replacement_ends_in_an_error_with_the_refused_file_in_place() {
            compare_runs(&[2], Refusal::Replacement(1));
        }
    }
}
