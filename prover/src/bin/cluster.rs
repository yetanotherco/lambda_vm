//! Process runner for the block levels over a configured TCP topology.

use std::{
    ffi::OsStr,
    fs::{self, OpenOptions},
    io::Write,
    os::unix::process::CommandExt,
    path::Path,
    process::{Command, ExitCode},
};

use lambda_vm_prover::{
    block_whir::{
        B3Rounds, BlockFormat, BlockOptions,
        cluster_tcp::{Topology, coordinate, serve},
    },
    lfm::{
        last_level::{LastLevel, last_level_file, last_wire},
        levels::{LevelStop, pin_family, prove_block_levels, verify_block_levels_file},
        proof::block_base_options_under,
        wlfm::WlfmFamily,
    },
};
use multilinear::whir_chain::GroupFold;

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("cluster: {e}");
            ExitCode::FAILURE
        }
    }
}

fn run() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<_> = std::env::args_os().skip(1).collect();
    let (family, command) = match args.as_slice() {
        [flag, family, command @ ..] if flag == "--family" => {
            let family = match family.to_str() {
                Some("poseidon" | "p1") => WlfmFamily::P1,
                Some("blake3" | "b3") => WlfmFamily::B3,
                _ => return Err("--family must be poseidon or blake3".into()),
            };
            (family, command)
        }
        _ => (WlfmFamily::P1, args.as_slice()),
    };
    let format = pin_family(
        BlockFormat::pinned_with(GroupFold::Indexed),
        family,
        B3Rounds::Fri,
    )?;
    let opts = block_base_options_under(&format.zf);
    match command {
        [mode, topology, id] if mode == "worker" => {
            let topology = Topology::read(Path::new(topology))?;
            let id = id
                .to_str()
                .ok_or("worker index must be a number")?
                .parse()?;
            let cards = topology.worker_cards(id)?;
            if cfg!(feature = "cuda") && std::env::var("LAMBDA_VM_CARDS").as_deref() != Ok(&cards) {
                // CUDA and Rayon read their environment at initialization.
                return Err(Command::new(std::env::current_exe()?)
                    .args(&args)
                    .env("LAMBDA_VM_CARDS", cards)
                    .exec()
                    .into());
            }
            serve(&topology, id, &opts, &format)?;
        }
        [mode, topology, elf, input, output] if mode == "prove" => {
            let options = BlockOptions::production();
            let topology = Topology::read(Path::new(topology))?;
            let elf = fs::read(elf)?;
            let input = if input == OsStr::new("-") {
                Vec::new()
            } else {
                fs::read(input)?
            };
            let run = coordinate(&topology, || {
                prove_block_levels(
                    &elf,
                    &input,
                    &opts,
                    &format,
                    &options,
                    family,
                    LevelStop::DEFAULT,
                    true,
                    true,
                )
            })?;
            let last = run
                .proofs
                .last()
                .ok_or("the levels produced no final proof")?;
            let shipped = last_level_file(last, LastLevel::pinned(family), last_wire()?)?;
            let statement = run.base_proof.statement();
            verify_block_levels_file(&elf, statement, family, &shipped)?;
            let file = rkyv::to_bytes::<rkyv::rancor::Error>(&(statement.to_owned(), shipped))?;
            publish(Path::new(output), &file)?;
        }
        _ => {
            return Err(
                "usage: cluster [--family poseidon|blake3] worker TOPOLOGY INDEX | \
                 cluster [--family poseidon|blake3] prove TOPOLOGY ELF INPUT|- OUTPUT"
                    .into(),
            );
        }
    }
    Ok(())
}

fn publish(output: &Path, bytes: &[u8]) -> std::io::Result<()> {
    let mut name = output.file_name().unwrap_or_default().to_os_string();
    name.push(format!(".{}.tmp", std::process::id()));
    let temporary = output.with_file_name(name);
    let mut file = OpenOptions::new()
        .create_new(true)
        .write(true)
        .open(&temporary)?;
    let result = file
        .write_all(bytes)
        .and_then(|()| fs::rename(&temporary, output));
    if result.is_err() {
        let _ = fs::remove_file(temporary);
    }
    result
}
