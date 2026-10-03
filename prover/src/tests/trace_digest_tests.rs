//! Prints a digest of every table of a few whole-run builds (rows sorted for the
//! HashMap-ordered tables), to compare two revisions of the trace builder.

use executor::elf::Elf;
use executor::vm::execution::Executor;
use stark::trace::TraceTable;

use crate::tables::MaxRowsConfig;
use crate::tables::trace_builder::Traces;
use crate::tables::types::{GoldilocksExtension, GoldilocksField};
use crate::test_utils::asm_elf_bytes;

type Table = TraceTable<GoldilocksField, GoldilocksExtension>;

fn digest(t: &Table, hashed: bool) -> String {
    let mut rows: Vec<Vec<u64>> = (0..t.main_table.height)
        .map(|r| {
            t.main_table
                .get_row(r)
                .iter()
                .map(|v| v.canonical())
                .collect()
        })
        .collect();
    if hashed {
        rows.sort();
    }
    let mut h = blake3::Hasher::new();
    h.update(&(t.main_table.width as u64).to_le_bytes());
    for row in rows {
        for v in row {
            h.update(&v.to_le_bytes());
        }
    }
    h.finalize().to_hex()[..16].to_string()
}

#[test]
#[ignore = "prints, for comparing revisions"]
fn trace_digests() {
    for name in [
        "sub",
        "all_instructions_64",
        "test_keccak",
        "test_keccak_multi",
    ] {
        for (label, max_rows) in [
            ("small", MaxRowsConfig::small()),
            ("default", MaxRowsConfig::default()),
        ] {
            let program = Elf::load(&asm_elf_bytes(name)).unwrap();
            let logs = Executor::new(&program, Vec::new())
                .unwrap()
                .run()
                .unwrap()
                .logs;
            let t = Traces::from_elf_and_logs(
                &program,
                &logs,
                &max_rows,
                &[],
                #[cfg(feature = "disk-spill")]
                stark::storage_mode::StorageMode::Ram,
            )
            .unwrap();
            let lists: [(&str, &Vec<Table>, bool); 21] = [
                ("CPU", &t.cpus, false),
                ("MEMW_R", &t.memw_registers, false),
                ("MEMW_A", &t.memw_aligneds, false),
                ("MEMW", &t.memws, false),
                ("LOAD", &t.loads, false),
                ("STORE", &t.stores, false),
                ("SHIFT", &t.shifts, false),
                ("CPU32", &t.cpu32s, false),
                ("COMMIT", &t.commits, false),
                ("KECCAK", &t.keccaks, false),
                ("KECCAK_RND", &t.keccak_rnds, false),
                ("ECSM", &t.ecsms, false),
                ("ECDAS", &t.ecdases, false),
                ("HINT", &t.hints, false),
                ("PAGE", &t.pages, false),
                ("LT", &t.lts, true),
                ("MUL", &t.muls, true),
                ("DVRM", &t.dvrms, true),
                ("BRANCH", &t.branches, true),
                ("EQ", &t.eqs, true),
                ("BYTEWISE", &t.bytewises, true),
            ];
            for (n, list, hashed) in lists {
                for (i, table) in list.iter().enumerate() {
                    println!("DIGEST {name} {label} {n}[{i}] {}", digest(table, hashed));
                }
            }
            for (n, table) in [
                ("BITWISE", &t.bitwise),
                ("DECODE", &t.decode),
                ("REGISTER", &t.register),
                ("HALT", &t.halt),
                ("KECCAK_RC", &t.keccak_rc),
                ("BLAKE3", &t.blake3),
            ] {
                println!("DIGEST {name} {label} {n} {}", digest(table, false));
            }
        }
    }
}
