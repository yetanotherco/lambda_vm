//! The measurement-only register-shape probe (`crate::regs_probe`): with the
//! knob off the CPU row is today's; the probe's CPU row is the format's (24
//! interactions in today's 24 GKR slots, five more columns). The two block
//! tests are box-only and each runs in a process of its own: one with the knob
//! unset, one with `LAMBDA_VM_PROBE_REGS_SHAPE=1` (the knob is read once a
//! process).

use crate::regs_probe;
use crate::tables::cpu::{
    self, bus_interactions, cols, num_columns, num_columns_for, probe_cols, probe_interactions,
};
use crate::tables::types::BusId;
use multilinear::whir_chain::GkrPack;
use stark::lookup::BusInteraction;
use stark::multilinear_table::batched::{ArgueShape, chunking};

/// The GKR input slots a table of `2^num_vars` rows and these interactions
/// takes under the production packing.
fn slots(interactions: &[BusInteraction], num_vars: usize) -> usize {
    let n = interactions.len();
    let shape = ArgueShape {
        num_vars,
        input_vars: multilinear::logup::input_layer_vars(n, num_vars),
        degree: 2,
        shifted: false,
        interactions: n,
    };
    let c = chunking(&shape, GkrPack::PRODUCTION);
    c.count << c.slot_vars
}

/// The columns an interaction reads.
fn columns_read(i: &BusInteraction) -> Vec<usize> {
    i.columns_read()
}

#[test]
fn with_the_knob_off_the_cpu_row_is_todays() {
    assert!(!regs_probe::on());
    assert_eq!(num_columns(), cols::NUM_COLUMNS);
    assert_eq!(num_columns_for(false), 37);
    for ts_base in [0, 4 << 21] {
        let today = bus_interactions(ts_base);
        assert!(today == cpu::today_interactions(ts_base));
        assert_eq!(today.len(), 20);
        let buses: Vec<u64> = today.iter().map(|i| i.bus_id).collect();
        let want: Vec<u64> = [
            BusId::Decode,
            BusId::Alu,
            BusId::Cpu32,
            BusId::Memw,
            BusId::Memw,
            BusId::Memw,
            BusId::MemoryOp,
            BusId::Memory,
            BusId::Memory,
            BusId::Memory,
            BusId::Memory,
            BusId::Branch,
            BusId::AreBytes,
            BusId::AreBytes,
            BusId::AreBytes,
            BusId::IsHalfword,
            BusId::IsHalfword,
            BusId::IsHalfword,
            BusId::IsHalfword,
            BusId::Ecall,
        ]
        .into_iter()
        .map(u64::from)
        .collect();
        assert_eq!(buses, want);
        assert!(
            today
                .iter()
                .all(|i| columns_read(i).iter().all(|&c| c < cols::NUM_COLUMNS))
        );
    }
    let air = crate::test_utils::create_cpu_air(
        &stark::proof::options::ProofOptions::default_test_options(),
        0,
    );
    assert_eq!(stark::traits::AIR::bus_interactions(&air).len(), 20);
}

#[test]
fn the_probes_cpu_row_takes_todays_gkr_slots() {
    let today = cpu::today_interactions(0);
    let probe = probe_interactions(0);
    assert_eq!(probe.len(), 24);
    for num_vars in [21, 20, 10] {
        assert_eq!(
            slots(&probe, num_vars),
            slots(&today, num_vars),
            "2^{num_vars} rows"
        );
    }
    // The block's full CPU chunk: three trees of eight, 24 slots, as today.
    assert_eq!(slots(&probe, 21), 24);
    // One more interaction would take 32.
    let mut more = probe.clone();
    more.push(probe[0].clone());
    assert_eq!(slots(&more, 21), 32);
}

#[test]
fn the_probe_keeps_todays_thirteen_and_reads_its_five_columns() {
    let today = cpu::today_interactions(0);
    let probe = probe_interactions(0);
    let kept: Vec<&BusInteraction> = today
        .iter()
        .filter(|i| i.bus_id != u64::from(BusId::Memw) && i.bus_id != u64::from(BusId::Memory))
        .collect();
    assert_eq!(kept.len(), 13);
    assert!(probe[..13].iter().collect::<Vec<_>>() == kept);
    // Six tokens, three range checks on the slots, two PC tokens.
    let added = &probe[13..];
    let on = |bus: BusId| added.iter().filter(|i| i.bus_id == u64::from(bus)).count();
    assert_eq!(on(BusId::Memory), 8);
    assert_eq!(on(BusId::IsHalfword), 3);
    // Senders: three consumed tokens, three range checks, the PC's consumed
    // token; receivers: three produced tokens and the PC's.
    assert_eq!(added.iter().filter(|i| i.is_sender).count(), 7);
    assert_eq!(added.iter().filter(|i| !i.is_sender).count(), 4);
    let width = num_columns_for(true);
    assert_eq!(width, 42);
    let read: std::collections::BTreeSet<usize> = added.iter().flat_map(columns_read).collect();
    for c in [
        probe_cols::PREV_TS_1,
        probe_cols::PREV_TS_2,
        probe_cols::PREV_TS_D,
        probe_cols::OLD_RD_0,
        probe_cols::OLD_RD_1,
    ] {
        assert!(read.contains(&c), "column {c} is read");
    }
    assert!(probe.iter().flat_map(columns_read).all(|c| c < width));
    // Every token carries four values: register, timestamp, two words.
    for i in added
        .iter()
        .filter(|i| i.bus_id == u64::from(BusId::Memory))
    {
        assert_eq!(i.values.len(), 4);
    }
}

/// The card's fill reads the same knob and adds the same zero columns, and
/// adds none with the knob off.
#[test]
fn the_card_fill_reads_the_probes_knob_and_columns() {
    assert_eq!(math_cuda::fill::REGS_PROBE_ENV, regs_probe::ENV);
    assert_eq!(
        math_cuda::fill::CPU_PROBE_COLUMNS,
        regs_probe::EXTRA_CPU_COLUMNS
    );
    assert_eq!(math_cuda::fill::cpu_extra_columns(), 0);
    assert_eq!(
        math_cuda::fill::CPU_COLUMNS + math_cuda::fill::CPU_PROBE_COLUMNS,
        num_columns_for(true)
    );
}

#[cfg(feature = "parallel")]
mod block {
    use crate::block_whir::{
        self, BlockFormat, BlockOptions, Deviations, prove_block_whir_with, verify_block_whir,
    };
    use crate::regs_probe;
    use crate::tables::MaxRowsConfig;
    use crate::test_utils::asm_elf_bytes;
    use crate::zf_format::ZfFormat;
    use multilinear::whir_chain::ArgueFormat;
    use stark::proof::options::ProofOptions;

    /// The production format (`block_whir_tests`' one group).
    fn format() -> BlockFormat {
        BlockFormat {
            zf: ZfFormat::DEFAULT,
            group_polys: block_whir::BLOCK_GROUP_POLYS,
            max_groups: block_whir::BLOCK_MAX_GROUPS,
            prepared: true,
            argue: ArgueFormat::PerTable,
            group: multilinear::whir_chain::GroupFold::PerPoly,
            leaf_scale: crate::lfm::whir_block::LeafScale::Fixed,
            leaf_query: crate::lfm::whir_chain::LeafQuery::V1,
            keccak_rnd: crate::tables::keccak_rnd::KeccakRndForm::Full,
        }
    }

    /// Streamed, chunks of 2^10 rows: `loop_4096` makes four full CPU chunks,
    /// held as their logs where the card fills them.
    fn options() -> BlockOptions {
        BlockOptions {
            max_rows: MaxRowsConfig::uniform(1 << 10),
            keccak_rnd_rows_log2: 3,
            ecdas_rows_log2: block_whir::BLOCK_ECDAS_ROWS_LOG2,
            keccak_rows_log2: block_whir::BLOCK_KECCAK_ROWS_LOG2,
            ecsm_rows_log2: block_whir::BLOCK_ECSM_ROWS_LOG2,
            copy_rows_log2: block_whir::BLOCK_COPY_ROWS_LOG2,
            drop_levels: multilinear::whir_commit::TreeDrop::uniform(3),
            window_log2: Some(9),
            drop_streamed_ops: false,
            layout_workers: 0,
            layout_ahead: Some(2),
            narrow: stark::multilinear_block::Narrowing::Card { min_cells: 0 },
            upload_ahead: true,
            memlog: false,
            finish_keccak_rnd_chunks: true,
            finish_cuts: true,
            finish_stream: true,
            rest_layout_bytes: Some(1 << 20),
            rest_build_bytes: None,
            pack_finished: true,
            gpack: true,
            regen: None,
            phase_b_overlap: stark::multilinear_block::PhaseBOverlap::Off,
        }
    }

    /// `loop_4096` proved as a block: its table counts and whether it verifies.
    fn prove_loop() -> (crate::TableCounts, bool) {
        let elf = asm_elf_bytes("loop_4096");
        let opts = ProofOptions::default_test_options();
        let (proof, _) = prove_block_whir_with(
            &elf,
            &[],
            &opts,
            &format(),
            &options(),
            &Deviations::default(),
        )
        .expect("prove");
        let verified = verify_block_whir(&proof, &elf, &opts, &format()).unwrap_or(false);
        eprintln!(
            "REGS PROBE BLOCK: probe {} · CPU tables {} · MEMW_R tables {} · MEMW_A tables {} · \
             verified {verified}",
            regs_probe::on(),
            proof.table_counts.cpu,
            proof.table_counts.memw_register,
            proof.table_counts.memw_aligned,
        );
        (proof.table_counts, verified)
    }

    /// Box, knob unset: `loop_4096` (about two register accesses a cycle)
    /// makes more MEMW_R tables than CPU tables, and its proof verifies.
    #[test]
    #[ignore = "box: proves a block; run alone with the knob unset"]
    fn without_the_register_shape_probe_a_block_keeps_its_register_rows() {
        assert!(!regs_probe::on(), "run with {} unset", regs_probe::ENV);
        let (counts, verified) = prove_loop();
        assert!(counts.memw_register >= counts.cpu, "{counts:?}");
        assert!(verified);
    }

    /// Box, `LAMBDA_VM_PROBE_REGS_SHAPE=1`: the CPU rows' register accesses
    /// make no MEMW_R row (what is left fits one table), the block proves,
    /// and its verifier refuses it.
    #[test]
    #[ignore = "box: proves a block; run alone with LAMBDA_VM_PROBE_REGS_SHAPE=1"]
    fn under_the_register_shape_probe_a_block_proves_without_its_cpu_register_rows() {
        assert!(regs_probe::on(), "run with {}=1", regs_probe::ENV);
        let (counts, verified) = prove_loop();
        assert!(counts.cpu >= 4, "{counts:?}");
        assert!(counts.memw_register <= 1, "{counts:?}");
        assert!(!verified, "a probe's proof must never verify");
    }
}
