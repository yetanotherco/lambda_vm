//! Manufacture a proof that differs from another in ONE named field:
//! `tamper_proof <IN.proof> <OUT.proof> <none|main-root|ood|counts> [TABLE]`.
//!
//! The control half of `cmp_proofs`. That tool reports `DIFFERING TABLES: N of
//! M` and exits non-zero, and until a pair that MUST differ has been seen to
//! trip it, a `0 of M` from it means nothing.
//!
//! ⛔ WHY NOT A BYTE FLIP. Flipping bytes at uniform offsets in the file does
//! not work, and that was measured rather than guessed: of 32 evenly spaced
//! flips in a 10.8 MB proof, 30 changed nothing `cmp_proofs` reads and 2 were
//! refused by rkyv validation. A proof that size is almost entirely query
//! openings and FRI decommitments, and the compared set is the roots, the OOD
//! evaluations, the final poly, the declared counts and a few lengths — a
//! small target the shotgun never hits. So the difference is made HERE, in the
//! deserialized proof, in a field named on the command line.
//!
//! `none` is the honest control and is not optional: it round-trips the proof
//! through deserialize and serialize WITHOUT touching it, so the pair proves
//! the round trip itself moves nothing. Without that arm, a re-serialization
//! artefact would masquerade as the manufactured trip and "the gate can fail"
//! would have been demonstrated by the wrong mechanism.
//!
//! Every arm asserts that the field it names actually changed, and refuses
//! rather than writing an unmodified proof: a tamper that quietly did nothing
//! would come back MATCH and read as a gate that cannot fail.
//!
//! Exit: 0 written, 2 usage, 4 the mutation did not apply.

use std::os::unix::fs::FileExt;

use prover::VmProof;

fn read(path: &str) -> VmProof {
    let file = std::fs::File::open(path).expect("open");
    let len = file.metadata().expect("metadata").len() as usize;
    let mut buf = rkyv::util::AlignedVec::<16>::with_capacity(len);
    buf.resize(len, 0);
    file.read_exact_at(&mut buf, 0).expect("read");
    rkyv::from_bytes::<VmProof, rkyv::rancor::Error>(&buf).expect("deserialize")
}

fn hex8(bytes: &[u8]) -> String {
    bytes
        .iter()
        .take(8)
        .map(|b| format!("{b:02x}"))
        .collect::<String>()
}

fn main() -> std::process::ExitCode {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 4 || args.len() > 5 {
        eprintln!("usage: tamper_proof <IN.proof> <OUT.proof> <none|main-root|ood|counts> [TABLE]");
        return std::process::ExitCode::from(2);
    }
    let (input, output, what) = (&args[1], &args[2], args[3].as_str());
    let table: usize = match args.get(4).map(|s| s.parse::<usize>()) {
        None => 0,
        Some(Ok(i)) => i,
        Some(Err(e)) => {
            eprintln!("bad TABLE index: {e}");
            return std::process::ExitCode::from(2);
        }
    };

    let mut proof = read(input);
    let tables = proof.proof.proofs.len();
    println!("TAMPER {what} on {input} -> {output}");
    println!("  tables: {tables}");
    if !matches!(what, "none" | "main-root" | "ood" | "counts") {
        eprintln!("unknown mutation '{what}' (want none, main-root, ood or counts)");
        return std::process::ExitCode::from(2);
    }
    if what != "counts" && what != "none" && table >= tables {
        eprintln!("table {table} is out of range: this proof holds {tables}");
        return std::process::ExitCode::from(4);
    }

    match what {
        // The control. Nothing is touched; the round trip is the whole point.
        "none" => {
            println!("  NOTHING TOUCHED — this is the round-trip control");
        }
        "main-root" => {
            let root = &mut proof.proof.proofs[table].lde_trace_main_merkle_root;
            let before = *root;
            root[0] ^= 0x01;
            let after = *root;
            if before == after {
                eprintln!("  the main root did not change");
                return std::process::ExitCode::from(4);
            }
            println!(
                "  table {table} MAIN ROOT {}… -> {}…",
                hex8(&before),
                hex8(&after)
            );
        }
        // Swap two entries that differ, so the OOD table changes without this
        // example needing to name a field type or build an element.
        "ood" => {
            let ood = &mut proof.proof.proofs[table].trace_ood_evaluations;
            let (h, w) = (ood.height, ood.width);
            println!("  table {table} OOD is {h} x {w}");
            let mut anchor: Option<(usize, usize)> = None;
            let mut swapped: Option<((usize, usize), (usize, usize))> = None;
            'scan: for r in 0..h {
                for c in 0..w {
                    match anchor {
                        None => anchor = Some((r, c)),
                        Some((r0, c0)) => {
                            if ood.get(r, c) != ood.get(r0, c0) {
                                let a = *ood.get(r0, c0);
                                let b = *ood.get(r, c);
                                ood.set(r0, c0, b);
                                ood.set(r, c, a);
                                swapped = Some(((r0, c0), (r, c)));
                                break 'scan;
                            }
                        }
                    }
                }
            }
            match swapped {
                Some((from, to)) => println!("  swapped OOD {from:?} with {to:?}"),
                None => {
                    eprintln!("  every OOD entry of table {table} is equal: nothing to swap");
                    return std::process::ExitCode::from(4);
                }
            }
        }
        // The declared layout, which is compared as a whole and has its own
        // line in the gate.
        "counts" => {
            let before = proof.table_counts.cpu;
            proof.table_counts.cpu = before + 1;
            println!("  table_counts.cpu {before} -> {}", proof.table_counts.cpu);
        }
        _ => unreachable!("the match above rejects every other value"),
    }

    let bytes = rkyv::to_bytes::<rkyv::rancor::Error>(&proof).expect("serialize");
    std::fs::write(output, &bytes).expect("write");
    println!("  wrote {} bytes to {output}", bytes.len());
    std::process::ExitCode::SUCCESS
}
