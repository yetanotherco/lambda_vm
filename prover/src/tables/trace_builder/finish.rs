//! The table phase of a build (phases 3–5) as a plan and its tables.
//!
//! [`FinishPlan::new`] runs what needs the whole run before any table: LT's
//! segments (phase 3), the BITWISE histogram (phase 4), HALT and the REGISTER
//! final state, the public output and the PAGE layout; and counts every table
//! it will build without generating one ([`RestHeader`], what a statement is
//! read from). [`FinishPlan::emit_all`] then generates the tables (phase 5),
//! as [`super::build_traces`] always did, and refuses tables the header did
//! not count. The tables are the same as before the split, which the frozen
//! oracle (`finish_oracle`, test-only) checks.

use super::*;

/// What a statement reads off a build before any of its tables exists.
pub(crate) struct RestHeader {
    /// How many tables each counted chip makes ([`Traces::table_counts`] of the
    /// tables the plan builds).
    pub(crate) table_counts: crate::TableCounts,
    /// One a PAGE table, in their order (none in a continuation epoch).
    pub(crate) page_configs: Vec<PageConfig>,
    pub(crate) public_output_bytes: Vec<u8>,
    pub(crate) num_blake3_ops: usize,
}

impl RestHeader {
    /// [`Traces::runtime_page_ranges`] of the tables the plan builds.
    // Read by the tests now and by the streamed finish once it lays the rest
    // out from the header.
    #[cfg_attr(not(test), allow(dead_code))]
    pub(crate) fn runtime_page_ranges(&self) -> Vec<crate::RuntimePageRange> {
        runtime_page_ranges_of(&self.page_configs)
    }
}

/// [`crate::TableCounts`] field by field, to compare two.
pub(super) fn counts_of(c: &crate::TableCounts) -> [usize; 21] {
    [
        c.cpu,
        c.lt,
        c.memw,
        c.memw_aligned,
        c.load,
        c.mul,
        c.dvrm,
        c.shift,
        c.branch,
        c.memw_register,
        c.eq,
        c.bytewise,
        c.store,
        c.cpu32,
        c.keccak,
        c.keccak_rnd,
        c.ecsm,
        c.ecdas,
        c.hint,
        c.commit,
        c.blake3,
    ]
}

/// LT's ops as phase 3 leaves them: the segments its list is the
/// concatenation of, in this order ([`Segmented`]); with
/// [`StreamSkip::concat_lt`], all of them in `walk`.
#[derive(Default)]
struct LtLists {
    /// The walk's LT ops not streamed and what DVRM implies, in blocks.
    walk: BlockVec<LtOperation>,
    /// The phase-3 LT ops of the MEMW and MEMW_A ops a windowed build dropped.
    memw_lt: CompactLt,
    /// The phase-3 LT ops of the MEMW ops left.
    from_memw: Vec<LtOperation>,
    memw_aligned_lt: CompactLt,
    from_memw_aligned: Vec<LtOperation>,
    /// HINT's range checks.
    from_hints: Vec<LtOperation>,
}

impl LtLists {
    fn segments(&self) -> Segmented<'_, LtOperation> {
        let mut parts: Vec<Part<'_, LtOperation>> = self.walk.parts().map(Part::Ops).collect();
        parts.extend([
            Part::Compact(&self.memw_lt),
            Part::Ops(&self.from_memw),
            Part::Compact(&self.memw_aligned_lt),
            Part::Ops(&self.from_memw_aligned),
            Part::Ops(&self.from_hints),
        ]);
        Segmented { parts }
    }
}

/// How many tables a chunked list of `len` ops makes at `max` ops a table, the
/// first `skip` of them streamed ahead: [`chunk_and_generate_skipping`]'s and
/// [`chunk_and_generate_segmented`]'s count (`tails`: the list holds only the
/// ops past the streamed chunks; `optional`: an empty list makes no table).
fn chunked(
    len: usize,
    max: usize,
    skip: usize,
    tails: bool,
    optional: bool,
) -> Result<usize, Error> {
    let chunks = len.div_ceil(max.max(1));
    if skip == 0 {
        return Ok(if len == 0 && !optional { 1 } else { chunks });
    }
    if tails {
        return Ok(skip + chunks);
    }
    if skip > chunks {
        return Err(Error::Prover(format!(
            "{skip} chunks were streamed but the run has {chunks} of this table"
        )));
    }
    Ok(chunks)
}

/// How many tables [`cut_tables`] makes of `n` ops at `rows` rows a table.
fn cuts(n: usize, rows: usize) -> usize {
    if n == 0 {
        return 0;
    }
    let total = n.next_power_of_two().max(4);
    if total <= rows { 1 } else { total / rows }
}

/// A build's table phase, planned: every list phase 5 reads, what phases 3 and
/// 4 derived from them, and the [`RestHeader`]. See the module docs.
pub(super) struct FinishPlan<'a> {
    memory_state: &'a MemoryState,
    register_init: &'a [u32],
    decode_trace: TraceTable<GoldilocksField, GoldilocksExtension>,
    decode_pc_to_row: &'a decode::PcToRow,
    max_rows: &'a crate::tables::MaxRowsConfig,
    #[cfg(feature = "disk-spill")]
    storage_mode: StorageMode,
    l2g_memory_bookend: bool,
    skip: StreamSkip,
    cpu_ops: Vec<CpuOperation>,
    memw_ops: Vec<MemwOperation>,
    memw_aligned_ops: Vec<memw_aligned::AlignedRow>,
    memw_register_rows: Vec<RegRow>,
    load_ops: Vec<LoadOperation>,
    lt: LtLists,
    shift_ops: BlockVec<ShiftOperation>,
    branch_ops: BlockVec<BranchOperation>,
    mul_ops: BlockVec<(MulOperation, bool)>,
    dvrm_ops: BlockVec<(DvrmOperation, bool)>,
    commit_ops: BlockVec<CommitOperation>,
    keccak_ops: BlockVec<KeccakOperation>,
    blake3_ops: BlockVec<Blake3Operation>,
    blake3_absorb_ops: BlockVec<blake3::Blake3AbsorbOperation>,
    eq_ops: BlockVec<eq::EqOperation>,
    bytewise_ops: BlockVec<bytewise::BytewiseOperation>,
    store_ops: BlockVec<store::StoreOperation>,
    cpu32_ops: BlockVec<cpu32::Cpu32Operation>,
    ecsm_ops: BlockVec<ecsm::EcsmOperation>,
    ecdas_ops: BlockVec<ecdas::EcdasOperation>,
    hint_ops: BlockVec<hint::HintOperation>,
    bitwise_histogram: bitwise::BitwiseHistogram,
    num_padding_rows: usize,
    halt_timestamp: u64,
    halt_next_pc: u64,
    register_final_state: FinalRegisterStateMap,
    header: RestHeader,
}

impl<'a> FinishPlan<'a> {
    /// Phases 3 and 4 and the header, from routed ops ([`super::build_traces`]'s
    /// arguments). `initial_image` controls PAGE: `Some(image)` plans real PAGE
    /// tables and their BITWISE lookups seeded from the initial-memory image;
    /// `None` plans none.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn new<I: ImageSource + Sync>(
        ops: CollectedOps,
        initial_image: Option<&I>,
        memory_state: &'a MemoryState,
        register_init: &'a [u32],
        decode_trace: TraceTable<GoldilocksField, GoldilocksExtension>,
        decode_pc_to_row: &'a decode::PcToRow,
        mut register_state: RegisterState,
        max_rows: &'a crate::tables::MaxRowsConfig,
        #[cfg(feature = "disk-spill")] storage_mode: StorageMode,
        private_input: &[u8],
        is_final: bool,
        l2g_memory_bookend: bool,
        skip: &StreamSkip,
        mut pre: Option<PreCounted>,
    ) -> Result<Self, Error> {
        let CollectedOps {
            cpu_ops,
            memw_ops,
            memw_aligned_ops,
            memw_register_rows,
            load_ops,
            lt_ops: lt_walk,
            shift_ops,
            bitwise_ops,
            branch_ops,
            mul_ops,
            dvrm_ops,
            commit_ops,
            keccak_ops,
            blake3_ops,
            blake3_absorb_ops,
            eq_ops,
            bytewise_ops,
            store_ops,
            cpu32_ops,
            ecsm_ops,
            ecdas_ops,
            hint_ops,
        } = ops;

        // =====================================================================
        // PHASE 3: MEMW → LT (timestamp ordering and overflow checks)
        // =====================================================================
        // LT's list is these segments in this order, kept apart: concatenating them
        // into one list reallocated multi-GiB copies at a block's finish.
        let memw_lt = pre
            .as_mut()
            .map(|pre| std::mem::take(&mut pre.memw_lt))
            .unwrap_or_default();
        let lt_from_memw = collect_lt_from_memw(&memw_ops[skip.memw_lt_done.min(memw_ops.len())..]);
        build_stamps::mark("p3a lt from memw");
        let memw_aligned_lt = pre
            .as_mut()
            .map(|pre| std::mem::take(&mut pre.memw_aligned_lt))
            .unwrap_or_default();
        let lt_from_memw_aligned = collect_lt_from_memw_aligned(
            &memw_aligned_ops[skip.memw_aligned_lt_done.min(memw_aligned_ops.len())..],
        );
        build_stamps::mark("p3b lt from memw_a");
        // HINT range-checks: selector < 3 and both address low limbs < 2^32 - 31 (matching
        // the executor's HintUnknownSelector / HintAddressOverflow rejections). Three LT ops
        // per hint call; the HINT table sends the matching ALU LT interactions.
        let lt_from_hints: Vec<LtOperation> = hint_ops
            .iter()
            .flat_map(|op| {
                [
                    LtOperation::new(op.hint_id, hint::HINT_SELECTOR_BOUND, false),
                    LtOperation::new(op.in_addr & 0xFFFF_FFFF, hint::HINT_ADDR_LIMB_BOUND, false),
                    LtOperation::new(op.out_addr & 0xFFFF_FFFF, hint::HINT_ADDR_LIMB_BOUND, false),
                ]
            })
            .collect();
        let lt = if skip.concat_lt {
            // The A arm: one list, each segment appended and then freed, as before.
            let mut all = lt_walk.into_vec();
            memw_lt.expand_into(0, memw_lt.len(), &mut all);
            drop(memw_lt);
            all.extend(lt_from_memw);
            memw_aligned_lt.expand_into(0, memw_aligned_lt.len(), &mut all);
            drop(memw_aligned_lt);
            all.extend(lt_from_memw_aligned);
            all.extend(lt_from_hints);
            LtLists {
                walk: BlockVec::from_vec(all),
                ..LtLists::default()
            }
        } else {
            LtLists {
                walk: lt_walk,
                memw_lt,
                from_memw: lt_from_memw,
                memw_aligned_lt,
                from_memw_aligned: lt_from_memw_aligned,
                from_hints: lt_from_hints,
            }
        };

        build_stamps::mark("p3 lt");

        // =====================================================================
        // PHASE 4: All → Bitwise lookups
        // =====================================================================
        #[cfg(feature = "instruments")]
        let __sp = stark::instruments::span("p4_bitwise_collect");

        let public_output_bytes: Vec<u8> = commit_ops
            .iter()
            .filter(|op| !op.end)
            .map(|op| op.value)
            .collect();

        // CPU padding rows send ARE_BYTES with all-zero values.
        // Add corresponding ops so the bitwise table multiplicities balance.
        // Without the streamed chunks' ops (`skip.tails`), their full chunks pad too.
        let padding = |len: usize| len.next_power_of_two().max(4) - len;
        let num_padding_rows: usize = cpu_ops
            .chunks(max_rows.cpu)
            .map(|chunk| padding(chunk.len()))
            .sum::<usize>()
            + if skip.tails {
                skip.cpu * padding(max_rows.cpu)
            } else {
                0
            };

        let streamed_last_ecall = pre.as_ref().and_then(|pre| pre.last_ecall);
        let bitwise_histogram = {
            let lt_ops = lt.segments();
            // The per-source bitwise collectors are all pure functions of their inputs, and the
            // BITWISE multiplicities are order-independent (they ride a permutation-invariant bus),
            // so every source can be collected in parallel and the per-worker histograms summed in
            // any order.
            //
            // MUL/DVRM dedup their per-unique bit-gated lookups PER CHIP INSTANCE, so pass the same
            // chunk size used to split them into instances so multiplicities match the per-instance
            // sends. MEMW_R sends IS_HALFWORD[timestamp_0 - old_timestamp_lo - 1]. PAGE does a
            // batched ARE_BYTES[init, fini] per row (skipped in continuation epochs, which the L2G
            // table owns). COMMIT sends AreBytes+IsHalfword; KECCAK_RND sends XOR/AND/ARE_BYTES/HWSL;
            // HINT sends ARE_BYTES for its 32 output cells.
            // We never concatenate the lookups into one giant `Vec<BitwiseOperation>` (~140 M ops /
            // ~560 MB at 10-tx whose only consumer is the multiplicity count). Each collector bumps
            // the `BitwiseHistogram` it is handed: the heavy sources (MEMW_R one-per-row, PAGE
            // one-per-byte, padding) count directly with no per-source Vec at all, and the small
            // sources fold their transient `collect_*` Vec in and drop it. The histogram is a
            // commutative monoid, so per-worker histograms tree-reduce to multiplicities that are
            // independent of accumulation order.
            type Collector<'a> = Box<dyn Fn(&mut bitwise::BitwiseHistogram) + Sync + 'a>;
            let mul_chunk = max_rows.mul;
            let dvrm_chunk = max_rows.dvrm;
            // The sources that are a sum over their ops are cut into slices of whole ops, so phase 4's
            // buckets share them (a whole source was one bucket's long pole) and no slice's list of
            // lookups grows with the run. MUL and DVRM deduplicate per instance, so each of their
            // slices is one instance. The in-walk lookups and MEMW_R are split in the parallel path
            // below; the rest stay one collector each.
            let mut collectors: Vec<Collector> = Vec::new();
            if p4_sliced() {
                // LT's segments slice by slice; a compact one through a temporary of a
                // slice's ops at a time.
                let lt_slice = p4_slice_len(P4Source::Lt, mul_chunk, dvrm_chunk);
                for part in &lt_ops.parts {
                    match *part {
                        Part::Ops(ops) => {
                            for slice in ops.chunks(lt_slice) {
                                collectors.push(Box::new(move |h| {
                                    h.add_ops(&collect_bitwise_from_lt(slice))
                                }));
                            }
                        }
                        Part::Compact(compact) => {
                            for start in (0..compact.len()).step_by(lt_slice) {
                                collectors.push(Box::new(move |h| {
                                    let mut slice = Vec::new();
                                    compact.expand_into(start, start + lt_slice, &mut slice);
                                    h.add_ops(&collect_bitwise_from_lt(&slice))
                                }));
                            }
                        }
                    }
                }
                // MUL and DVRM deduplicate per instance: their slices are the
                // instances of the concatenation (copied only where one straddles two
                // blocks). The other sources are sums over their ops: block by block.
                let mul_slice = p4_slice_len(P4Source::Mul, mul_chunk, dvrm_chunk);
                for k in 0..mul_ops.len().div_ceil(mul_slice) {
                    let mul_ops = &mul_ops;
                    collectors.push(Box::new(move |h| {
                        let slice = mul_ops.range(k * mul_slice, (k + 1) * mul_slice);
                        h.add_ops(&collect_bitwise_from_mul(&slice, mul_chunk))
                    }));
                }
                let dvrm_slice = p4_slice_len(P4Source::Dvrm, mul_chunk, dvrm_chunk);
                for k in 0..dvrm_ops.len().div_ceil(dvrm_slice) {
                    let dvrm_ops = &dvrm_ops;
                    collectors.push(Box::new(move |h| {
                        let slice = dvrm_ops.range(k * dvrm_slice, (k + 1) * dvrm_slice);
                        h.add_ops(&collect_bitwise_from_dvrm(&slice, dvrm_chunk))
                    }));
                }
                let branch_slice = p4_slice_len(P4Source::Branch, mul_chunk, dvrm_chunk);
                for slice in branch_ops
                    .parts()
                    .flat_map(|part| part.chunks(branch_slice))
                {
                    collectors.push(Box::new(move |h| {
                        h.add_ops(&collect_bitwise_from_branch(slice))
                    }));
                }
                let shift_slice = p4_slice_len(P4Source::Shift, mul_chunk, dvrm_chunk);
                for slice in shift_ops.parts().flat_map(|part| part.chunks(shift_slice)) {
                    collectors.push(Box::new(move |h| {
                        h.add_ops(&shift::collect_bitwise_from_shift(slice))
                    }));
                }
                let bytewise_slice = p4_slice_len(P4Source::Bytewise, mul_chunk, dvrm_chunk);
                for slice in bytewise_ops
                    .parts()
                    .flat_map(|part| part.chunks(bytewise_slice))
                {
                    collectors.push(Box::new(move |h| {
                        for op in slice {
                            h.add_ops(&op.collect_bitwise_ops());
                        }
                    }));
                }
                let eq_slice = p4_slice_len(P4Source::Eq, mul_chunk, dvrm_chunk);
                for slice in eq_ops.parts().flat_map(|part| part.chunks(eq_slice)) {
                    collectors.push(Box::new(move |h| {
                        for op in slice {
                            h.add_ops(&op.collect_bitwise_ops());
                        }
                    }));
                }
                let store_slice = p4_slice_len(P4Source::Store, mul_chunk, dvrm_chunk);
                for slice in store_ops.parts().flat_map(|part| part.chunks(store_slice)) {
                    collectors.push(Box::new(move |h| {
                        for op in slice {
                            h.add_ops(&op.collect_bitwise_ops());
                        }
                    }));
                }
                for slice in memw_aligned_ops.chunks(p4_slice_len(
                    P4Source::MemwAligned,
                    mul_chunk,
                    dvrm_chunk,
                )) {
                    collectors.push(Box::new(move |h| {
                        h.add_ops(&collect_bitwise_from_memw_aligned(slice))
                    }));
                }
                let keccak_slice = p4_slice_len(P4Source::Keccak, mul_chunk, dvrm_chunk);
                for slice in keccak_ops
                    .parts()
                    .flat_map(|part| part.chunks(keccak_slice))
                {
                    collectors.push(Box::new(move |h| {
                        h.add_ops(&collect_bitwise_from_keccak(slice))
                    }));
                }
                let ecdas_slice = p4_slice_len(P4Source::Ecdas, mul_chunk, dvrm_chunk);
                for slice in ecdas_ops.parts().flat_map(|part| part.chunks(ecdas_slice)) {
                    collectors.push(Box::new(move |h| {
                        h.add_ops(&collect_bitwise_from_ecdas(slice))
                    }));
                }
            } else {
                // `LAMBDA_VM_P4_SLICED=0`: every source one collector, as before the slices (the A/B's
                // control). The same multiplicities: the histogram is a commutative sum.
                collectors.extend([
                    Box::new(|h: &mut bitwise::BitwiseHistogram| {
                        for part in &lt_ops.parts {
                            match *part {
                                Part::Ops(ops) => h.add_ops(&collect_bitwise_from_lt(ops)),
                                Part::Compact(compact) => {
                                    for start in (0..compact.len()).step_by(1 << 20) {
                                        let mut slice = Vec::new();
                                        compact.expand_into(start, start + (1 << 20), &mut slice);
                                        h.add_ops(&collect_bitwise_from_lt(&slice));
                                    }
                                }
                            }
                        }
                    }) as Collector,
                    // MUL and DVRM deduplicate per instance: instance by instance.
                    Box::new(|h| {
                        for k in 0..mul_ops.len().div_ceil(mul_chunk.max(1)) {
                            let slice = mul_ops.range(k * mul_chunk, (k + 1) * mul_chunk);
                            h.add_ops(&collect_bitwise_from_mul(&slice, mul_chunk))
                        }
                    }),
                    Box::new(|h| {
                        for k in 0..dvrm_ops.len().div_ceil(dvrm_chunk.max(1)) {
                            let slice = dvrm_ops.range(k * dvrm_chunk, (k + 1) * dvrm_chunk);
                            h.add_ops(&collect_bitwise_from_dvrm(&slice, dvrm_chunk))
                        }
                    }),
                    Box::new(|h| {
                        for part in branch_ops.parts() {
                            h.add_ops(&collect_bitwise_from_branch(part))
                        }
                    }),
                    Box::new(|h| {
                        for part in shift_ops.parts() {
                            h.add_ops(&shift::collect_bitwise_from_shift(part))
                        }
                    }),
                    Box::new(|h| {
                        for op in bytewise_ops.iter() {
                            h.add_ops(&op.collect_bitwise_ops());
                        }
                    }),
                    Box::new(|h| {
                        for op in eq_ops.iter() {
                            h.add_ops(&op.collect_bitwise_ops());
                        }
                    }),
                    Box::new(|h| {
                        for op in store_ops.iter() {
                            h.add_ops(&op.collect_bitwise_ops());
                        }
                    }),
                    Box::new(|h| h.add_ops(&collect_bitwise_from_memw_aligned(&memw_aligned_ops))),
                    Box::new(|h| {
                        for part in keccak_ops.parts() {
                            h.add_ops(&collect_bitwise_from_keccak(part))
                        }
                    }),
                    Box::new(|h| {
                        for part in ecdas_ops.parts() {
                            h.add_ops(&collect_bitwise_from_ecdas(part))
                        }
                    }),
                ]);
            }
            collectors.extend([
                // Small lists, one block each at a block's size: whole.
                Box::new(|h: &mut bitwise::BitwiseHistogram| {
                    h.add_ops(&collect_bitwise_from_commit(&commit_ops.whole()))
                }) as Collector,
                Box::new(|h| {
                    if !strip_blake3_side_effects() {
                        h.add_ops(&collect_bitwise_from_blake3(
                            &blake3_ops.whole(),
                            &blake3_absorb_ops.whole(),
                        ));
                    }
                }),
                Box::new(|h| h.add_ops(&collect_bitwise_from_ecsm(&ecsm_ops.whole()))),
                Box::new(|h| h.add_ops(&collect_bitwise_from_hint(&hint_ops.whole()))),
                Box::new(|h| add_padding_byte_checks(h, num_padding_rows)),
            ]);
            if let Some(image) = initial_image
                && !l2g_memory_bookend
            {
                collectors.push(Box::new(move |h| {
                    collect_bitwise_from_page(image, memory_state, l2g_memory_bookend, h)
                }));
            }
            let (mut base, counted_iw, counted_reg) = match pre {
                Some(pre) => (pre.histogram, pre.bitwise_ops, pre.memw_register_rows),
                None => (bitwise::BitwiseHistogram::new(), 0, 0),
            };
            // The in-walk lookups past the counted ones, block by block.
            let mut skip_iw = counted_iw;
            let uncounted_iw: Vec<&[BitwiseOperation]> = bitwise_ops
                .parts()
                .filter_map(|part| {
                    let from = skip_iw.min(part.len());
                    skip_iw -= from;
                    (from < part.len()).then(|| &part[from..])
                })
                .collect();
            let uncounted_reg = &memw_register_rows[counted_reg..];

            #[cfg(feature = "parallel")]
            {
                use rayon::prelude::*;
                // Cap concurrent 80 MiB histograms at `cap` to bound peak memory. The two dominant
                // sources — the in-walk lookups and MEMW_R (each tens of millions of items) — are
                // split into ~`cap` row-range slices so they parallelize INTERNALLY instead of each
                // pinning one core while the rest idle. Every unit (whole collectors + the heavy
                // slices) is round-robined into exactly `cap` buckets, one histogram each, so the
                // split heavy work is spread across buckets rather than piled into one.
                // add_ops/bump/merge form a commutative monoid, so any partition yields
                // byte-identical multiplicities (same as the serial fallback below).
                let cap = rayon::current_num_threads().clamp(1, 8);
                let mut units: Vec<Collector> = Vec::with_capacity(collectors.len() + 2 * cap);
                let iw_total: usize = uncounted_iw.iter().map(|part| part.len()).sum();
                let iw_chunk = iw_total.div_ceil(cap).max(1);
                for slice in uncounted_iw.iter().flat_map(|part| part.chunks(iw_chunk)) {
                    units.push(Box::new(move |h| h.add_ops(slice)));
                }
                let reg_chunk = uncounted_reg.len().div_ceil(cap).max(1);
                for slice in uncounted_reg.chunks(reg_chunk) {
                    units.push(Box::new(move |h| {
                        memw_register::collect_bitwise_from_memw_register(slice, h)
                    }));
                }
                units.extend(collectors);

                let mut buckets: Vec<Vec<Collector>> = (0..cap).map(|_| Vec::new()).collect();
                for (i, unit) in units.into_iter().enumerate() {
                    buckets[i % cap].push(unit);
                }
                if let Some(reduced) = buckets
                    .par_iter()
                    .map(|bucket| {
                        let mut h = bitwise::BitwiseHistogram::new();
                        for f in bucket {
                            f(&mut h);
                        }
                        h
                    })
                    .reduce_with(|mut a, b| {
                        a.merge(&b);
                        a
                    })
                {
                    base.merge(&reduced);
                }
            }
            #[cfg(not(feature = "parallel"))]
            {
                for part in &uncounted_iw {
                    base.add_ops(part);
                }
                memw_register::collect_bitwise_from_memw_register(uncounted_reg, &mut base);
                for f in &collectors {
                    f(&mut base);
                }
            }
            base
        };
        // The in-walk lookups have been counted into the histogram; free them now.
        drop(bitwise_ops);
        #[cfg(feature = "instruments")]
        drop(__sp);
        build_stamps::mark("p4 bitwise");

        // A monolithic run or the final continuation epoch terminates on the program's
        // halt ECALL. Intermediate continuation epochs do not halt, so fall back to the
        // last cycle's timestamp and skip HALT-based PC finalization — the PC is carried
        // to the next epoch via the register snapshot, and HALT is excluded from the
        // proof (`include_halt = false`) so `halt_trace` is unused there.
        let (halt_timestamp, halt_next_pc) = if is_final {
            let (halt_timestamp, halt_next_pc) = cpu_ops
                .iter()
                .rev()
                .find(|op| op.decode.fields.ecall)
                .map(|op| (op.timestamp, op.next_pc))
                .or(streamed_last_ecall)
                .ok_or(Error::MissingHaltEcall)?;
            // Finalize the PC (x255) on the REGISTER table. The CPU padding rows carry
            // pc=1 and chain the inline-PC `memory` tokens with a +4 timestamp cadence
            // starting from the HALT chip's emit_pc at `halt_timestamp + 1`; the last
            // padding write therefore lands at `halt_timestamp + 4*num_padding_rows + 1`
            // (= `halt_timestamp + 1` when there is no padding). The REGISTER final token
            // must match that last write to balance the memory argument.
            register_state.write_pc(1, halt_timestamp + 4 * num_padding_rows as u64 + 1);
            (halt_timestamp, halt_next_pc)
        } else {
            (cpu_ops.last().map(|op| op.timestamp).unwrap_or(0), 0)
        };

        let register_final_state = register_state.to_final_state_map();

        // The PAGE layout: none in a continuation epoch (the L2G table owns every
        // touched cell's Memory init/fini there) or without an image.
        let page_configs = match initial_image {
            Some(image) if !l2g_memory_bookend => {
                page_configs_of(image, memory_state, private_input)
            }
            _ => Vec::new(),
        };
        let num_blake3_ops = blake3_ops.len() + blake3_absorb_ops.len();

        // Every table the plan builds, counted without building one.
        let tails = skip.tails;
        let keccak_rnd = match (keccak_ops.len(), skip.keccak_rnd_rows) {
            (n, 0) => usize::from(n > 0),
            (0, _) => 0,
            (n, rows) => {
                let total = (n * 24).next_power_of_two().max(4);
                let chunks = if total <= rows { 1 } else { total / rows };
                if skip.keccak_rnd > 0 && (total <= rows || skip.keccak_rnd > chunks) {
                    return Err(Error::Prover(format!(
                        "{} KECCAK_RND chunks were streamed but the run has {chunks}",
                        skip.keccak_rnd
                    )));
                }
                chunks
            }
        };
        let accelerator = |n: usize, rows: usize| {
            if rows > 0 {
                cuts(n, rows)
            } else {
                usize::from(n > 0)
            }
        };
        let table_counts = crate::TableCounts {
            cpu: chunked(cpu_ops.len(), max_rows.cpu, skip.cpu, tails, false)?,
            lt: chunked(lt.segments().len(), max_rows.lt, skip.lt, tails, true)?,
            memw: chunked(memw_ops.len(), max_rows.memw, skip.memw, tails, true)?,
            memw_aligned: chunked(
                memw_aligned_ops.len(),
                max_rows.memw_aligned,
                skip.memw_aligned,
                tails,
                true,
            )?,
            load: chunked(load_ops.len(), max_rows.load, skip.load, tails, true)?,
            mul: chunked(mul_ops.len(), max_rows.mul, 0, false, true)?,
            dvrm: chunked(dvrm_ops.len(), max_rows.dvrm, 0, false, true)?,
            shift: chunked(shift_ops.len(), max_rows.shift, skip.shift, tails, true)?,
            branch: chunked(branch_ops.len(), max_rows.branch, 0, false, true)?,
            memw_register: chunked(
                memw_register_rows.len(),
                max_rows.memw_register,
                skip.memw_register,
                tails,
                false,
            )?,
            eq: chunked(eq_ops.len(), max_rows.eq, 0, false, true)?,
            bytewise: chunked(bytewise_ops.len(), max_rows.bytewise, 0, false, true)?,
            store: chunked(store_ops.len(), max_rows.store, skip.store, tails, true)?,
            cpu32: chunked(cpu32_ops.len(), max_rows.cpu32, 0, false, true)?,
            keccak: accelerator(keccak_ops.len(), skip.keccak_rows),
            keccak_rnd,
            ecsm: accelerator(ecsm_ops.len(), skip.ecsm_rows),
            ecdas: accelerator(ecdas_ops.len(), skip.ecdas_rows),
            hint: usize::from(!hint_ops.is_empty()),
            commit: usize::from(!commit_ops.is_empty()),
            blake3: usize::from(num_blake3_ops > 0),
        };

        Ok(Self {
            memory_state,
            register_init,
            decode_trace,
            decode_pc_to_row,
            max_rows,
            #[cfg(feature = "disk-spill")]
            storage_mode,
            l2g_memory_bookend,
            skip: *skip,
            cpu_ops,
            memw_ops,
            memw_aligned_ops,
            memw_register_rows,
            load_ops,
            lt,
            shift_ops,
            branch_ops,
            mul_ops,
            dvrm_ops,
            commit_ops,
            keccak_ops,
            blake3_ops,
            blake3_absorb_ops,
            eq_ops,
            bytewise_ops,
            store_ops,
            cpu32_ops,
            ecsm_ops,
            ecdas_ops,
            hint_ops,
            bitwise_histogram,
            num_padding_rows,
            halt_timestamp,
            halt_next_pc,
            register_final_state,
            header: RestHeader {
                table_counts,
                page_configs,
                public_output_bytes,
                num_blake3_ops,
            },
        })
    }

    /// What a statement reads off the build, before any table is built.
    // Read by the tests now and by the streamed finish once it lays the rest
    // out from the header.
    #[cfg_attr(not(test), allow(dead_code))]
    pub(crate) fn header(&self) -> &RestHeader {
        &self.header
    }

    /// The header's counts replaced (a test's wrong plan, which
    /// [`Self::emit_all`] must refuse).
    #[cfg(test)]
    pub(super) fn with_counts(mut self, table_counts: crate::TableCounts) -> Self {
        self.header.table_counts = table_counts;
        self
    }

    /// Phase 5: every table, as [`super::build_traces`] always built them, in
    /// one parallel scope; refused unless they are the ones the header counted.
    pub(super) fn emit_all(self) -> Result<Traces, Error> {
        let Self {
            memory_state,
            register_init,
            decode_trace,
            decode_pc_to_row,
            max_rows,
            #[cfg(feature = "disk-spill")]
            storage_mode,
            l2g_memory_bookend,
            skip,
            cpu_ops,
            memw_ops,
            memw_aligned_ops,
            memw_register_rows,
            load_ops,
            lt,
            shift_ops,
            branch_ops,
            mul_ops,
            dvrm_ops,
            commit_ops,
            keccak_ops,
            blake3_ops,
            blake3_absorb_ops,
            eq_ops,
            bytewise_ops,
            store_ops,
            cpu32_ops,
            ecsm_ops,
            ecdas_ops,
            hint_ops,
            bitwise_histogram,
            num_padding_rows,
            halt_timestamp,
            halt_next_pc,
            register_final_state,
            header,
        } = self;
        let lt_ops = lt.segments();

        // =====================================================================
        // PHASE 5: Generate final traces (parallelized)
        // =====================================================================
        #[cfg(feature = "instruments")]
        let __sp = stark::instruments::span("p5_generate_tables");

        // Each build below reads disjoint op lists and writes its own table, so
        // they all run in one rayon scope. Disk-spill stays sequential: its
        // generate→spill order keeps trace memory bounded.
        let cpu_ops_ref = &cpu_ops;
        let pack = skip.pack;
        // A packing build writes each table packed as it generates it where the
        // generator can (G-pack, `StreamSkip::gpack`); `packed_by` then finds it
        // packed.
        #[cfg(feature = "disk-spill")]
        let gpack = pack && skip.gpack && storage_mode != StorageMode::Disk;
        #[cfg(not(feature = "disk-spill"))]
        let gpack = pack && skip.gpack;
        let form = if gpack {
            TraceForm::Narrow
        } else {
            TraceForm::Wide
        };
        let gen_cpus = || {
            chunk_and_generate_skipping(
                cpu_ops_ref,
                max_rows.cpu,
                skip.cpu,
                skip.tails,
                false,
                packed_by(pack, |ops| cpu::generate_cpu_trace_as(ops, form)),
                #[cfg(feature = "disk-spill")]
                storage_mode,
            )
        };
        let gen_memws = || {
            chunk_and_generate_skipping(
                &memw_ops,
                max_rows.memw,
                skip.memw,
                skip.tails,
                true,
                packed_by(pack, |ops| memw::generate_memw_trace_as(ops, form)),
                #[cfg(feature = "disk-spill")]
                storage_mode,
            )
        };
        let gen_memw_aligneds = || {
            chunk_and_generate_skipping(
                &memw_aligned_ops,
                max_rows.memw_aligned,
                skip.memw_aligned,
                skip.tails,
                true,
                packed_by(pack, |ops| {
                    memw_aligned::generate_memw_aligned_trace_as(ops, form)
                }),
                #[cfg(feature = "disk-spill")]
                storage_mode,
            )
        };
        let gen_memw_registers = || {
            // Direct-to-column fill from compact RegRows — the register fast path never
            // materializes a `Vec<MemwOperation>`.
            chunk_and_generate_skipping(
                &memw_register_rows,
                max_rows.memw_register,
                skip.memw_register,
                skip.tails,
                false,
                packed_by(pack, |ops| {
                    memw_register::generate_memw_register_trace_from_rows_as(ops, form)
                }),
                #[cfg(feature = "disk-spill")]
                storage_mode,
            )
        };
        let gen_loads = || {
            chunk_and_generate_skipping(
                &load_ops,
                max_rows.load,
                skip.load,
                skip.tails,
                true,
                packed_by(pack, |ops| load::generate_load_trace_as(ops, form)),
                #[cfg(feature = "disk-spill")]
                storage_mode,
            )
        };
        let gen_lts = || {
            chunk_and_generate_segmented(
                &lt_ops,
                max_rows.lt,
                skip.lt,
                skip.tails,
                true,
                packed_by(pack, |ops| lt::generate_lt_trace_as(ops, form)),
                #[cfg(feature = "disk-spill")]
                storage_mode,
            )
        };
        let gen_shifts = || {
            chunk_and_generate_segmented(
                &shift_ops.segments(),
                max_rows.shift,
                skip.shift,
                skip.tails,
                true,
                packed_by(pack, |ops| shift::generate_shift_trace_as(ops, form)),
                #[cfg(feature = "disk-spill")]
                storage_mode,
            )
        };
        let gen_muls = || {
            chunk_and_generate_segmented(
                &mul_ops.segments(),
                max_rows.mul,
                0,
                false,
                true,
                packed_by(pack, |ops| mul::generate_mul_trace_as(ops, form)),
                #[cfg(feature = "disk-spill")]
                storage_mode,
            )
        };
        let gen_dvrms = || {
            chunk_and_generate_segmented(
                &dvrm_ops.segments(),
                max_rows.dvrm,
                0,
                false,
                true,
                packed_by(pack, |ops| dvrm::generate_dvrm_trace_as(ops, form)),
                #[cfg(feature = "disk-spill")]
                storage_mode,
            )
        };
        let gen_branches = || {
            chunk_and_generate_segmented(
                &branch_ops.segments(),
                max_rows.branch,
                0,
                false,
                true,
                packed_by(pack, |ops| branch::generate_branch_trace_as(ops, form)),
                #[cfg(feature = "disk-spill")]
                storage_mode,
            )
        };
        // Auxiliary ALU / memory / CPU32 dispatch chips, each filtered out of the CPU
        // ops above.
        let gen_eqs = || {
            chunk_and_generate_segmented::<eq::EqOperation>(
                &eq_ops.segments(),
                max_rows.eq,
                0,
                false,
                true,
                packed_by(pack, |ops| eq::generate_eq_trace_as(ops, form)),
                #[cfg(feature = "disk-spill")]
                storage_mode,
            )
        };
        let gen_bytewises = || {
            chunk_and_generate_segmented::<bytewise::BytewiseOperation>(
                &bytewise_ops.segments(),
                max_rows.bytewise,
                0,
                false,
                true,
                packed_by(pack, |ops| bytewise::generate_bytewise_trace_as(ops, form)),
                #[cfg(feature = "disk-spill")]
                storage_mode,
            )
        };
        let gen_stores = || {
            chunk_and_generate_segmented(
                &store_ops.segments(),
                max_rows.store,
                skip.store,
                skip.tails,
                true,
                packed_by(pack, |ops| store::generate_store_trace_as(ops, form)),
                #[cfg(feature = "disk-spill")]
                storage_mode,
            )
        };
        let gen_cpu32s = || {
            chunk_and_generate_segmented::<cpu32::Cpu32Operation>(
                &cpu32_ops.segments(),
                max_rows.cpu32,
                0,
                false,
                true,
                packed_by(pack, |ops| cpu32::generate_cpu32_trace_as(ops, form)),
                #[cfg(feature = "disk-spill")]
                storage_mode,
            )
        };
        let gen_bitwise = || {
            let mut bitwise = bitwise::generate_bitwise_trace();
            // Fill the MU columns (11..=20) from the accumulated histogram.
            bitwise_histogram.fill_multiplicities(&mut bitwise);
            packed_if(pack, bitwise)
        };
        // Each CPU operation looks up the DECODE table once; padding rows look up
        // pc=1 (the CPU padding entry). When CPU is split, each chunk pads
        // independently.
        let gen_decode = move || {
            let mut decode = decode_trace;
            let mut decode_lookups: Vec<u64> = cpu_ops_ref.iter().map(|op| op.decode.pc).collect();
            decode_lookups.extend(std::iter::repeat_n(cpu::CPU_PADDING_PC, num_padding_rows));
            decode::update_multiplicities(&mut decode, decode_pc_to_row, &decode_lookups);
            packed_if(pack, decode)
        };
        let gen_commits = || {
            generate_optional(
                &commit_ops.whole(),
                packed_by(pack, |ops| commit::generate_commit_trace_as(ops, form)),
                #[cfg(feature = "disk-spill")]
                storage_mode,
            )
        };
        let gen_keccaks = || {
            if skip.keccak_rows > 0 {
                return cut_tables(
                    &keccak_ops,
                    skip.keccak_rows,
                    pack,
                    form,
                    keccak::rows_written_packed,
                    keccak::generate_keccak_rows_as,
                );
            }
            generate_optional(
                &keccak_ops.whole(),
                keccak::generate_keccak_trace,
                #[cfg(feature = "disk-spill")]
                storage_mode,
            )
        };
        let gen_keccak_rnds = || {
            let keccak_rnd_ops: Vec<KeccakRoundOperation> = keccak_ops
                .iter()
                .map(|op| KeccakRoundOperation {
                    timestamp: op.timestamp,
                    input: op.input,
                    output: op.output,
                })
                .collect();
            if skip.keccak_rnd_rows == 0 {
                // Whole, it may be cut after the build: it stays wide.
                return generate_optional(
                    &keccak_rnd_ops,
                    keccak_rnd::generate_keccak_rnd_trace,
                    #[cfg(feature = "disk-spill")]
                    storage_mode,
                );
            }
            keccak_rnd_chunks(
                &keccak_rnd_ops,
                skip.keccak_rnd_rows,
                skip.keccak_rnd,
                pack,
                form,
            )
        };
        let gen_blake3 = || {
            packed_if(
                pack,
                blake3::generate_blake3_trace(&blake3_ops.whole(), &blake3_absorb_ops.whole()),
            )
        };
        let gen_keccak_rc = || {
            let mut keccak_rc_trace = keccak_rc::generate_keccak_rc_trace();
            keccak_rc::update_multiplicities(&mut keccak_rc_trace, keccak_ops.len());
            packed_if(pack, keccak_rc_trace)
        };
        // The plan's page configs: none in a continuation epoch
        // (l2g_memory_bookend) or without an image ([`FinishPlan::new`]).
        let gen_pages = || -> Vec<_> {
            header
                .page_configs
                .iter()
                .map(|config| packed_if(pack, page_trace(config, memory_state, l2g_memory_bookend)))
                .collect()
        };
        let gen_register = || {
            packed_if(
                pack,
                register::generate_register_trace(&register_final_state, register_init),
            )
        };
        let gen_halt = || {
            packed_if(
                pack,
                halt::generate_halt_trace(halt_timestamp, halt_next_pc),
            )
        };
        // ECSM accelerator traces. A program that does not use ECSM carries no ECSM
        // and no ECDAS table at all — not a padded one.
        let gen_ecsms = || {
            if skip.ecsm_rows > 0 {
                return cut_tables(
                    &ecsm_ops,
                    skip.ecsm_rows,
                    pack,
                    form,
                    ecsm::rows_written_packed,
                    ecsm::generate_ecsm_rows_as,
                );
            }
            generate_optional(
                &ecsm_ops.whole(),
                ecsm::generate_ecsm_trace,
                #[cfg(feature = "disk-spill")]
                storage_mode,
            )
        };
        let gen_ecdases = || {
            if skip.ecdas_rows > 0 {
                return cut_tables(
                    &ecdas_ops,
                    skip.ecdas_rows,
                    pack,
                    form,
                    ecdas::rows_written_packed,
                    ecdas::generate_ecdas_rows_as,
                );
            }
            generate_optional(
                &ecdas_ops.whole(),
                ecdas::generate_ecdas_trace,
                #[cfg(feature = "disk-spill")]
                storage_mode,
            )
        };
        // HINT table. Absent entirely for programs that make no hint ecalls.
        let gen_hints = || {
            generate_optional(
                &hint_ops.whole(),
                packed_by(pack, |ops| hint::generate_hint_trace_as(ops, form)),
                #[cfg(feature = "disk-spill")]
                storage_mode,
            )
        };

        let (mut cpus_slot, mut memws_slot, mut memw_aligneds_slot, mut memw_registers_slot) =
            (None, None, None, None);
        let (mut loads_slot, mut lts_slot, mut shifts_slot, mut muls_slot) =
            (None, None, None, None);
        let (mut dvrms_slot, mut branches_slot, mut bitwise_slot, mut decode_slot) =
            (None, None, None, None);
        let (mut commits_slot, mut keccaks_slot, mut keccak_rnds_slot, mut keccak_rc_slot) =
            (None, None, None, None);
        let mut blake3_slot = None;
        let (mut pages_slot, mut register_slot, mut halt_slot) = (None, None, None);
        let (mut eqs_slot, mut bytewises_slot, mut stores_slot, mut cpu32s_slot) =
            (None, None, None, None);
        let (mut ecsms_slot, mut ecdases_slot) = (None, None);
        let mut hints_slot = None;

        #[cfg(feature = "disk-spill")]
        let sequential = storage_mode == StorageMode::Disk || cfg!(not(feature = "parallel"));
        #[cfg(not(feature = "disk-spill"))]
        let sequential = cfg!(not(feature = "parallel"));

        if !sequential {
            #[cfg(feature = "parallel")]
            rayon::scope(|s| {
                macro_rules! spawn_into {
                    ($slot:ident, $gen:ident) => {{
                        let slot = &mut $slot;
                        s.spawn(move |_| {
                            *slot = Some($gen());
                            build_stamps::mark(concat!("p5 ", stringify!($gen)));
                        });
                    }};
                }
                // Heaviest builds first so the scheduler overlaps them with the rest.
                spawn_into!(memw_registers_slot, gen_memw_registers);
                spawn_into!(cpus_slot, gen_cpus);
                spawn_into!(memws_slot, gen_memws);
                spawn_into!(lts_slot, gen_lts);
                spawn_into!(decode_slot, gen_decode);
                spawn_into!(branches_slot, gen_branches);
                spawn_into!(bitwise_slot, gen_bitwise);
                spawn_into!(muls_slot, gen_muls);
                spawn_into!(memw_aligneds_slot, gen_memw_aligneds);
                spawn_into!(loads_slot, gen_loads);
                spawn_into!(shifts_slot, gen_shifts);
                spawn_into!(dvrms_slot, gen_dvrms);
                spawn_into!(pages_slot, gen_pages);
                spawn_into!(keccaks_slot, gen_keccaks);
                spawn_into!(keccak_rnds_slot, gen_keccak_rnds);
                spawn_into!(keccak_rc_slot, gen_keccak_rc);
                spawn_into!(blake3_slot, gen_blake3);
                spawn_into!(commits_slot, gen_commits);
                spawn_into!(register_slot, gen_register);
                spawn_into!(halt_slot, gen_halt);
                spawn_into!(eqs_slot, gen_eqs);
                spawn_into!(bytewises_slot, gen_bytewises);
                spawn_into!(stores_slot, gen_stores);
                spawn_into!(cpu32s_slot, gen_cpu32s);
                spawn_into!(ecsms_slot, gen_ecsms);
                spawn_into!(ecdases_slot, gen_ecdases);
                spawn_into!(hints_slot, gen_hints);
            });
        } else {
            cpus_slot = Some(gen_cpus());
            memws_slot = Some(gen_memws());
            memw_aligneds_slot = Some(gen_memw_aligneds());
            memw_registers_slot = Some(gen_memw_registers());
            loads_slot = Some(gen_loads());
            lts_slot = Some(gen_lts());
            shifts_slot = Some(gen_shifts());
            muls_slot = Some(gen_muls());
            dvrms_slot = Some(gen_dvrms());
            branches_slot = Some(gen_branches());
            bitwise_slot = Some(gen_bitwise());
            decode_slot = Some(gen_decode());
            commits_slot = Some(gen_commits());
            keccaks_slot = Some(gen_keccaks());
            keccak_rnds_slot = Some(gen_keccak_rnds());
            keccak_rc_slot = Some(gen_keccak_rc());
            blake3_slot = Some(gen_blake3());
            pages_slot = Some(gen_pages());
            register_slot = Some(gen_register());
            halt_slot = Some(gen_halt());
            eqs_slot = Some(gen_eqs());
            bytewises_slot = Some(gen_bytewises());
            stores_slot = Some(gen_stores());
            cpu32s_slot = Some(gen_cpu32s());
            ecsms_slot = Some(gen_ecsms());
            ecdases_slot = Some(gen_ecdases());
            hints_slot = Some(gen_hints());
        }

        build_stamps::mark("p5 end");
        const PHASE5_RAN: &str = "phase 5 generation ran in one of the branches above";
        let cpus = cpus_slot.expect(PHASE5_RAN)?;
        let memws = memws_slot.expect(PHASE5_RAN)?;
        let memw_aligneds = memw_aligneds_slot.expect(PHASE5_RAN)?;
        let memw_registers = memw_registers_slot.expect(PHASE5_RAN)?;
        let loads = loads_slot.expect(PHASE5_RAN)?;
        let lts = lts_slot.expect(PHASE5_RAN)?;
        let shifts = shifts_slot.expect(PHASE5_RAN)?;
        let muls = muls_slot.expect(PHASE5_RAN)?;
        let dvrms = dvrms_slot.expect(PHASE5_RAN)?;
        let branches = branches_slot.expect(PHASE5_RAN)?;
        let eqs = eqs_slot.expect(PHASE5_RAN)?;
        let bytewises = bytewises_slot.expect(PHASE5_RAN)?;
        let stores = stores_slot.expect(PHASE5_RAN)?;
        let cpu32s = cpu32s_slot.expect(PHASE5_RAN)?;
        #[allow(unused_mut)]
        let mut bitwise = bitwise_slot.expect(PHASE5_RAN);
        #[allow(unused_mut)]
        let mut decode = decode_slot.expect(PHASE5_RAN);
        let commits = commits_slot.expect(PHASE5_RAN)?;
        let keccaks = keccaks_slot.expect(PHASE5_RAN)?;
        let keccak_rnds = keccak_rnds_slot.expect(PHASE5_RAN)?;
        let keccak_rc_trace = keccak_rc_slot.expect(PHASE5_RAN);
        let blake3_trace = blake3_slot.expect(PHASE5_RAN);
        #[allow(unused_mut)]
        let mut pages = pages_slot.expect(PHASE5_RAN);
        #[allow(unused_mut)]
        let mut register_trace = register_slot.expect(PHASE5_RAN);
        #[allow(unused_mut)]
        let mut halt_trace = halt_slot.expect(PHASE5_RAN);
        let ecsms = ecsms_slot.expect(PHASE5_RAN)?;
        let ecdases = ecdases_slot.expect(PHASE5_RAN)?;
        let hints = hints_slot.expect(PHASE5_RAN)?;

        // Fixed-size and per-page tables aren't built through `chunk_and_generate`,
        // so spill them here before returning.
        #[cfg(feature = "disk-spill")]
        if storage_mode == StorageMode::Disk {
            bitwise
                .main_table
                .spill_to_disk()
                .map_err(|e| Error::Prover(format!("disk-spill bitwise: {e}")))?;
            decode
                .main_table
                .spill_to_disk()
                .map_err(|e| Error::Prover(format!("disk-spill decode: {e}")))?;
            register_trace
                .main_table
                .spill_to_disk()
                .map_err(|e| Error::Prover(format!("disk-spill register: {e}")))?;
            halt_trace
                .main_table
                .spill_to_disk()
                .map_err(|e| Error::Prover(format!("disk-spill halt: {e}")))?;
            for page in &mut pages {
                page.main_table
                    .spill_to_disk()
                    .map_err(|e| Error::Prover(format!("disk-spill page: {e}")))?;
            }
        }

        // Continuation callers derive the real cross-epoch boundary from this set and
        // install its L2G trace after provenance is applied. Avoid building a
        // throwaway genesis-only L2G trace here.
        let touched_memory_cells = if l2g_memory_bookend {
            touched_cells_from_memory_state(memory_state)
        } else {
            Vec::new()
        };
        let local_to_global = local_to_global::generate_local_to_global_trace(&[]);
        #[cfg(feature = "instruments")]
        drop(__sp);
        let RestHeader {
            table_counts,
            page_configs,
            public_output_bytes,
            num_blake3_ops,
        } = header;
        let traces = Traces {
            cpus,
            bitwise,
            lts,
            shifts,
            memws,
            memw_aligneds,
            loads,
            decode,
            muls,
            dvrms,
            pages,
            page_configs,
            register: register_trace,
            public_output_bytes,
            branches,
            halt: halt_trace,
            commits,
            keccaks,
            keccak_rnds,
            blake3: blake3_trace,
            num_blake3_ops,
            keccak_rc: keccak_rc_trace,
            ecsms,
            ecdases,
            hints,
            memw_registers,
            local_to_global,
            touched_memory_cells,
            eqs,
            bytewises,
            stores,
            cpu32s,
        };
        // The tables are the ones the header counted, or the build is refused: the
        // header is what a statement can be read from before the tables exist.
        let built = traces.table_counts();
        if counts_of(&built) != counts_of(&table_counts) {
            return Err(Error::Prover(format!(
                "the finish built {built:?}, its plan counted {table_counts:?}"
            )));
        }
        Ok(traces)
    }
}
