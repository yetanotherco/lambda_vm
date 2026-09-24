//! Compare two `VmProof` files table by table: `cmp_proofs A.proof B.proof`.
//!
//! A diagnostic for the streaming prover: which table, and which part of it,
//! first departs from the monolithic prover's proof of the same execution.

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

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let (a, b) = (read(&args[1]), read(&args[2]));
    println!(
        "tables: {} vs {}",
        a.proof.proofs.len(),
        b.proof.proofs.len()
    );
    println!(
        "table_counts equal: {}",
        format!("{:?}", a.table_counts) == format!("{:?}", b.table_counts)
    );
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
        if !diffs.is_empty() && shown < 40 {
            println!("table {i}: {}", diffs.join(", "));
            shown += 1;
        }
    }
    println!("(showing at most 40 differing tables)");
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
}
