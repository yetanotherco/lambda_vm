//! Compare two `VmProof` files table by table: `cmp_proofs A.proof B.proof`.
//!
//! A diagnostic for the streaming prover: which table, and which part of it,
//! first departs from the monolithic prover's proof of the same execution.
//!
//! Also a gate. It ends with `DIFFERING TABLES: N of M` and exits non-zero when
//! the proofs differ, when they hold different numbers of tables, or when they
//! declare different layouts — so a caller can branch on it. The grinding nonce
//! is not compared (it is not reproducible across processes); every field that
//! is compared is.
//!
//! Exit: 0 they match, 1 they differ, 2 bad usage.

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

fn main() -> std::process::ExitCode {
    let args: Vec<String> = std::env::args().collect();
    if args.len() != 3 {
        eprintln!("usage: cmp_proofs <A.proof> <B.proof>");
        return std::process::ExitCode::from(2);
    }
    let (a, b) = (read(&args[1]), read(&args[2]));
    let (na, nb) = (a.proof.proofs.len(), b.proof.proofs.len());
    let counts_equal = format!("{:?}", a.table_counts) == format!("{:?}", b.table_counts);
    println!("tables: {na} vs {nb}");
    println!("table_counts equal: {counts_equal}");
    println!("counts A: {:?}", a.table_counts);
    println!("counts B: {:?}", b.table_counts);
    println!(
        "runtime_page_ranges equal: {}",
        format!("{:?}", a.runtime_page_ranges) == format!("{:?}", b.runtime_page_ranges)
    );
    println!(
        "num_private_input_pages: {} vs {}",
        a.num_private_input_pages, b.num_private_input_pages
    );
    println!(
        "public_output equal: {}",
        a.public_output == b.public_output
    );
    let mut shown = 0;
    // Counted over every compared table, never capped. The print below is
    // capped; the count is what the caller gates on, and a capped count would
    // read as agreement past the fortieth difference.
    let mut differing = 0usize;
    for (i, (x, y)) in a.proof.proofs.iter().zip(b.proof.proofs.iter()).enumerate() {
        let mut diffs = Vec::new();
        if x.trace_length != y.trace_length {
            diffs.push(format!(
                "trace_length {} vs {}",
                x.trace_length, y.trace_length
            ));
        }
        if x.lde_trace_main_merkle_root != y.lde_trace_main_merkle_root {
            diffs.push("main root".into());
        }
        if x.lde_trace_precomputed_merkle_root != y.lde_trace_precomputed_merkle_root {
            diffs.push("precomputed root".into());
        }
        if x.lde_trace_aux_merkle_root != y.lde_trace_aux_merkle_root {
            diffs.push("aux root".into());
        }
        if format!("{:?}", x.trace_ood_evaluations) != format!("{:?}", y.trace_ood_evaluations) {
            diffs.push("ood".into());
        }
        if x.composition_poly_root != y.composition_poly_root {
            diffs.push("composition root".into());
        }
        if x.fri_layers_merkle_roots != y.fri_layers_merkle_roots {
            diffs.push("fri roots".into());
        }
        if x.fri_final_poly_coeffs != y.fri_final_poly_coeffs {
            diffs.push("fri final".into());
        }
        if x.query_list.len() != y.query_list.len() {
            diffs.push(format!(
                "queries {} vs {}",
                x.query_list.len(),
                y.query_list.len()
            ));
        }
        if !diffs.is_empty() {
            differing += 1;
            if shown < 40 {
                println!("table {i}: {}", diffs.join(", "));
                shown += 1;
            }
        }
    }
    if differing > shown {
        println!("({} further differing tables not shown)", differing - shown);
    }
    let main_diff: Vec<usize> = a
        .proof
        .proofs
        .iter()
        .zip(b.proof.proofs.iter())
        .enumerate()
        .filter(|(_, (x, y))| x.lde_trace_main_merkle_root != y.lde_trace_main_merkle_root)
        .map(|(i, _)| i)
        .collect();
    println!(
        "tables whose MAIN root differs ({}): {:?}",
        main_diff.len(),
        main_diff
    );

    // The line a caller gates on. `compared` is what the zip above actually
    // looked at, so it is never larger than the shorter proof.
    let compared = na.min(nb);
    println!("DIFFERING TABLES: {differing} of {compared}");

    // Tables present on one side only were never compared at all. Reporting
    // "0 of N" while silently dropping them is the failure this exists to stop.
    let lengths_equal = na == nb;
    if !lengths_equal {
        println!(
            "TABLE COUNT MISMATCH: {na} vs {nb} — {} table(s) on one side only, not compared",
            na.abs_diff(nb)
        );
    }
    if !counts_equal {
        println!("TABLE COUNTS MISMATCH: the two proofs declare different layouts");
    }

    let matched = differing == 0 && lengths_equal && counts_equal;
    println!("CMP_VERDICT: {}", if matched { "MATCH" } else { "DIFFER" });
    if matched {
        std::process::ExitCode::SUCCESS
    } else {
        std::process::ExitCode::FAILURE
    }
}
