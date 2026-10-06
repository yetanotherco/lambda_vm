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
#[derive(Clone)]
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
    branch_ops: CompactBranch,
    mul_ops: BlockVec<(MulOperation, bool)>,
    dvrm_ops: BlockVec<(DvrmOperation, bool)>,
    commit_ops: BlockVec<CommitOperation>,
    keccak_ops: BlockVec<KeccakOperation>,
    blake3_ops: BlockVec<Blake3Operation>,
    blake3_absorb_ops: BlockVec<blake3::Blake3AbsorbOperation>,
    eq_ops: CompactEq,
    bytewise_ops: CompactBytewise,
    store_ops: BlockVec<store::StoreOperation>,
    cpu32_ops: BlockVec<cpu32::Cpu32Operation>,
    ecsm_ops: BlockVec<ecsm::EcsmOperation>,
    ecdas_ops: BlockVec<ecdas::CompactEcdasOp>,
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
                for k in 0..branch_ops.len().div_ceil(branch_slice.max(1)) {
                    let branch_ops = &branch_ops;
                    collectors.push(Box::new(move |h| {
                        let slice = branch_ops.range(k * branch_slice, (k + 1) * branch_slice);
                        h.add_ops(&collect_bitwise_from_branch(&slice))
                    }));
                }
                let shift_slice = p4_slice_len(P4Source::Shift, mul_chunk, dvrm_chunk);
                for slice in shift_ops.parts().flat_map(|part| part.chunks(shift_slice)) {
                    collectors.push(Box::new(move |h| {
                        h.add_ops(&shift::collect_bitwise_from_shift(slice))
                    }));
                }
                let bytewise_slice = p4_slice_len(P4Source::Bytewise, mul_chunk, dvrm_chunk);
                for k in 0..bytewise_ops.len().div_ceil(bytewise_slice.max(1)) {
                    let bytewise_ops = &bytewise_ops;
                    collectors.push(Box::new(move |h| {
                        for op in bytewise_ops.range(k * bytewise_slice, (k + 1) * bytewise_slice) {
                            h.add_ops(&op.collect_bitwise_ops());
                        }
                    }));
                }
                let eq_slice = p4_slice_len(P4Source::Eq, mul_chunk, dvrm_chunk);
                for k in 0..eq_ops.len().div_ceil(eq_slice.max(1)) {
                    let eq_ops = &eq_ops;
                    collectors.push(Box::new(move |h| {
                        for op in eq_ops.range(k * eq_slice, (k + 1) * eq_slice) {
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
                        h.add_ops(&collect_bitwise_from_compact_ecdas(slice))
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
                        for k in 0..branch_ops.len().div_ceil(1 << 20) {
                            let slice = branch_ops.range(k << 20, (k + 1) << 20);
                            h.add_ops(&collect_bitwise_from_branch(&slice))
                        }
                    }),
                    Box::new(|h| {
                        for part in shift_ops.parts() {
                            h.add_ops(&shift::collect_bitwise_from_shift(part))
                        }
                    }),
                    Box::new(|h| {
                        for k in 0..bytewise_ops.len().div_ceil(1 << 20) {
                            for op in bytewise_ops.range(k << 20, (k + 1) << 20) {
                                h.add_ops(&op.collect_bitwise_ops());
                            }
                        }
                    }),
                    Box::new(|h| {
                        for k in 0..eq_ops.len().div_ceil(1 << 20) {
                            for op in eq_ops.range(k << 20, (k + 1) << 20) {
                                h.add_ops(&op.collect_bitwise_ops());
                            }
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
                            h.add_ops(&collect_bitwise_from_compact_ecdas(part))
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
                    |rows, num_rows, form| {
                        ecdas::generate_ecdas_rows_of(rows, num_rows, form, ecdas::widen)
                    },
                );
            }
            generate_optional(
                &ecdas_ops.whole(),
                |rows| {
                    let num_rows = rows.len().next_power_of_two().max(4);
                    ecdas::generate_ecdas_rows_of(rows, num_rows, TraceForm::Wide, ecdas::widen)
                },
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

type Table = TraceTable<GoldilocksField, GoldilocksExtension>;

/// A table of the rest, not yet built: building it is calling this.
type Job<'a> = Box<dyn FnOnce() -> Result<Table, Error> + Send + 'a>;

/// A one-row-an-op family's cut generator: (ops, rows, form) → table.
type CutGenerator<'a, T> =
    std::sync::Arc<dyn Fn(&[T], usize, TraceForm) -> Table + Send + Sync + 'a>;

/// A wave's slot: a unit's AIR position and, for a table, its number among
/// the rest's tables, its family and its job.
type WaveSlot<'a> = (usize, Option<(usize, usize, Job<'a>)>);

/// One table of the rest, in AIR order ([`FinishPlan::units`]).
pub(super) enum Unit<'a> {
    /// A streamed chunk's slot of a family ([`FAMILIES`]): its table went out
    /// with the windows.
    Streamed { family: usize },
    /// A table to build: its family ([`FAMILIES`]), an estimate of its rows and
    /// the job that builds it.
    Build {
        family: usize,
        rows: usize,
        job: Job<'a>,
    },
}

/// What the streamed finish hands on, in AIR order ([`FinishPlan::emit_streamed`]):
/// the table at AIR `position`, or `None` for a streamed chunk's slot.
pub(crate) struct Emitted {
    pub(crate) position: usize,
    pub(crate) table: Option<Table>,
}

/// The rest's families in AIR order (`VmAirs::air_refs`), each with its
/// columns at eight bytes a cell: what a family's first table is estimated
/// at, before one of them has been built.
const FAMILIES: [(&str, usize); 27] = [
    ("BITWISE", bitwise::cols::NUM_COLUMNS),
    ("DECODE", decode::cols::NUM_COLUMNS),
    ("KECCAK_RC", keccak_rc::cols::NUM_COLUMNS),
    ("REGISTER", register::cols::NUM_COLUMNS),
    ("HALT", halt::cols::NUM_COLUMNS),
    ("COMMIT", commit::cols::NUM_COLUMNS),
    ("KECCAK", keccak::cols::NUM_COLUMNS),
    ("KECCAK_RND", keccak_rnd::cols::NUM_COLUMNS),
    ("ECSM", ecsm::cols::NUM_COLUMNS),
    ("ECDAS", ecdas::cols::NUM_COLUMNS),
    ("HINT", hint::cols::NUM_COLUMNS),
    ("BLAKE3", blake3::cols::NUM_COLUMNS),
    ("CPU", cpu::cols::NUM_COLUMNS),
    ("LT", lt::cols::NUM_COLUMNS),
    ("SHIFT", shift::cols::NUM_COLUMNS),
    ("MEMW", memw::cols::NUM_COLUMNS),
    ("MEMW_A", memw_aligned::cols::NUM_COLUMNS),
    ("LOAD", load::cols::NUM_COLUMNS),
    ("MUL", mul::cols::NUM_COLUMNS),
    ("DVRM", dvrm::cols::NUM_COLUMNS),
    ("BRANCH", branch::cols::NUM_COLUMNS),
    ("PAGE", page::cols::NUM_COLUMNS),
    ("MEMW_R", memw_register::cols::NUM_COLUMNS),
    ("EQ", eq::cols::NUM_COLUMNS),
    ("BYTEWISE", bytewise::cols::NUM_COLUMNS),
    ("STORE", store::cols::NUM_COLUMNS),
    ("CPU32", cpu32::cols::NUM_COLUMNS),
];

/// [`FAMILIES`]' index of a family.
fn family(name: &str) -> usize {
    FAMILIES
        .iter()
        .position(|&(f, _)| f == name)
        .unwrap_or(usize::MAX)
}

/// The bytes a table's main trace takes: packed, or eight a cell.
fn table_bytes(table: &Table) -> usize {
    table.narrow_main().map_or_else(
        || table.main_table.width * table.main_table.height * std::mem::size_of::<u64>(),
        |packed| packed.data().len(),
    )
}

/// The rows a table of `n` ops is padded to, as most generators pad it: an
/// estimate only.
fn rows_of(n: usize) -> usize {
    n.next_power_of_two().max(4)
}

/// A chunked list's units ([`chunk_and_generate_skipping`]'s and
/// [`chunk_and_generate_segmented`]'s tables, in their order): `skip` slots
/// streamed ahead, then a job a chunk of `max` ops (`chunk(start, end)`), an
/// empty list making one empty table unless `optional`.
#[allow(clippy::too_many_arguments)]
fn chunked_units<'a>(
    family: usize,
    len: usize,
    max: usize,
    skip: usize,
    tails: bool,
    optional: bool,
    chunk: impl Fn(usize, usize) -> Job<'a>,
) -> Result<Vec<Unit<'a>>, Error> {
    let max = max.max(1);
    let ranges: Vec<(usize, usize)> = (0..len.div_ceil(max))
        .map(|k| (k * max, ((k + 1) * max).min(len)))
        .collect();
    let (placeholders, ranges) = if skip == 0 {
        let ranges = if len == 0 && !optional {
            vec![(0, 0)]
        } else {
            ranges
        };
        (0, ranges)
    } else if tails {
        (skip, ranges)
    } else {
        if skip > ranges.len() {
            return Err(Error::Prover(format!(
                "{skip} chunks were streamed but the run has {} of this table",
                ranges.len()
            )));
        }
        (skip, ranges[skip..].to_vec())
    };
    let mut units: Vec<Unit<'a>> = (0..placeholders)
        .map(|_| Unit::Streamed { family })
        .collect();
    units.extend(ranges.into_iter().map(|(start, end)| Unit::Build {
        family,
        rows: rows_of(end - start),
        job: chunk(start, end),
    }));
    Ok(units)
}

/// A one-row-an-op family's units ([`cut_tables`]'s tables): none without
/// ops, one table when the whole is no taller than `rows`, else its cuts of
/// `rows` rows; or, without cuts (`rows` 0), the whole table wide
/// (`whole`), as the finish builds it to be split later.
fn cut_units<'a, T: Clone + Send + Sync + 'a>(
    family: usize,
    ops: &std::sync::Arc<BlockVec<T>>,
    rows: usize,
    pack: bool,
    form: TraceForm,
    generate: CutGenerator<'a, T>,
    whole: impl FnOnce(std::sync::Arc<BlockVec<T>>) -> Job<'a>,
) -> Result<Vec<Unit<'a>>, Error> {
    let n = ops.len();
    if n == 0 {
        return Ok(Vec::new());
    }
    if rows == 0 {
        return Ok(vec![Unit::Build {
            family,
            rows: rows_of(n),
            job: whole(std::sync::Arc::clone(ops)),
        }]);
    }
    if !rows.is_power_of_two() {
        return Err(Error::Prover(format!(
            "tables cut at {rows} rows: a power of two is needed"
        )));
    }
    let total = rows_of(n);
    if total <= rows {
        let ops = std::sync::Arc::clone(ops);
        return Ok(vec![Unit::Build {
            family,
            rows: total,
            job: Box::new(move || Ok(packed_if(pack, generate(&ops.whole(), total, form)))),
        }]);
    }
    Ok((0..total / rows)
        .map(|k| {
            let ops = std::sync::Arc::clone(ops);
            let generate = std::sync::Arc::clone(&generate);
            let (first, end) = ((k * rows).min(n), ((k + 1) * rows).min(n));
            Unit::Build {
                family,
                rows,
                job: Box::new(move || {
                    Ok(packed_if(
                        pack,
                        generate(&ops.range(first, end), rows, form),
                    ))
                }) as Job<'a>,
            }
        })
        .collect())
}

/// `job` if `n > 0`: one table, or none ([`generate_optional`]'s tables).
fn optional_unit<'a>(family: usize, n: usize, job: impl FnOnce() -> Job<'a>) -> Vec<Unit<'a>> {
    if n == 0 {
        return Vec::new();
    }
    vec![Unit::Build {
        family,
        rows: rows_of(n),
        job: job(),
    }]
}

impl<'a> FinishPlan<'a> {
    /// Phase 5 as units: every table the header counted (and BITWISE, DECODE,
    /// KECCAK_RC, REGISTER and HALT), in AIR order, each a job that builds it
    /// as [`Self::emit_all`] does, or a streamed chunk's slot. A family's list
    /// is held by its jobs alone and goes with the last of them. Refused unless
    /// the units are the tables the header counted.
    pub(super) fn units(self) -> Result<Vec<Unit<'a>>, Error> {
        use std::sync::Arc;
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
        #[cfg(feature = "disk-spill")]
        if storage_mode == StorageMode::Disk {
            return Err(Error::Prover(
                "the streamed finish holds its tables in memory".into(),
            ));
        }
        let pack = skip.pack;
        #[cfg(feature = "disk-spill")]
        let gpack = pack && skip.gpack && storage_mode != StorageMode::Disk;
        #[cfg(not(feature = "disk-spill"))]
        let gpack = pack && skip.gpack;
        let form = if gpack {
            TraceForm::Narrow
        } else {
            TraceForm::Wide
        };
        let tails = skip.tails;
        let mut units: Vec<Unit<'a>> = Vec::new();
        let cpu_ops = Arc::new(cpu_ops);

        // BITWISE, DECODE, KECCAK_RC, REGISTER, HALT.
        units.push(Unit::Build {
            family: family("BITWISE"),
            rows: bitwise::NUM_ROWS,
            job: Box::new(move || {
                let mut bitwise = bitwise::generate_bitwise_trace();
                bitwise_histogram.fill_multiplicities(&mut bitwise);
                Ok(packed_if(pack, bitwise))
            }),
        });
        {
            let cpu_ops = Arc::clone(&cpu_ops);
            units.push(Unit::Build {
                family: family("DECODE"),
                rows: decode_trace.num_rows(),
                job: Box::new(move || {
                    let mut decode = decode_trace;
                    let mut lookups: Vec<u64> = cpu_ops.iter().map(|op| op.decode.pc).collect();
                    lookups.extend(std::iter::repeat_n(cpu::CPU_PADDING_PC, num_padding_rows));
                    decode::update_multiplicities(&mut decode, decode_pc_to_row, &lookups);
                    Ok(packed_if(pack, decode))
                }),
            });
        }
        let num_keccak = keccak_ops.len();
        units.push(Unit::Build {
            family: family("KECCAK_RC"),
            rows: 64,
            job: Box::new(move || {
                let mut keccak_rc = keccak_rc::generate_keccak_rc_trace();
                keccak_rc::update_multiplicities(&mut keccak_rc, num_keccak);
                Ok(packed_if(pack, keccak_rc))
            }),
        });
        units.push(Unit::Build {
            family: family("REGISTER"),
            rows: 64,
            job: Box::new(move || {
                Ok(packed_if(
                    pack,
                    register::generate_register_trace(&register_final_state, register_init),
                ))
            }),
        });
        units.push(Unit::Build {
            family: family("HALT"),
            rows: 4,
            job: Box::new(move || {
                Ok(packed_if(
                    pack,
                    halt::generate_halt_trace(halt_timestamp, halt_next_pc),
                ))
            }),
        });

        // COMMIT, KECCAK, KECCAK_RND, ECSM, ECDAS, HINT, BLAKE3.
        let commit_ops = Arc::new(commit_ops);
        units.extend(optional_unit(family("COMMIT"), commit_ops.len(), || {
            Box::new(move || {
                Ok(packed_if(
                    pack,
                    commit::generate_commit_trace_as(&commit_ops.whole(), form),
                ))
            })
        }));
        let keccak_ops = Arc::new(keccak_ops);
        units.extend(cut_units(
            family("KECCAK"),
            &keccak_ops,
            skip.keccak_rows,
            pack,
            form,
            Arc::new(keccak::generate_keccak_rows_as),
            |ops| Box::new(move || Ok(keccak::generate_keccak_trace(&ops.whole()))),
        )?);
        units.extend(keccak_rnd_units(&keccak_ops, &skip, pack, form)?);
        drop(keccak_ops);
        let ecsm_ops = Arc::new(ecsm_ops);
        units.extend(cut_units(
            family("ECSM"),
            &ecsm_ops,
            skip.ecsm_rows,
            pack,
            form,
            Arc::new(ecsm::generate_ecsm_rows_as),
            |ops| Box::new(move || Ok(ecsm::generate_ecsm_trace(&ops.whole()))),
        )?);
        drop(ecsm_ops);
        let ecdas_ops = Arc::new(ecdas_ops);
        units.extend(cut_units(
            family("ECDAS"),
            &ecdas_ops,
            skip.ecdas_rows,
            pack,
            form,
            Arc::new(|rows: &[ecdas::CompactEcdasOp], num_rows, form| {
                ecdas::generate_ecdas_rows_of(rows, num_rows, form, ecdas::widen)
            }),
            |ops| {
                Box::new(move || {
                    let rows = ops.whole();
                    let num_rows = rows_of(rows.len());
                    Ok(ecdas::generate_ecdas_rows_of(
                        &rows,
                        num_rows,
                        TraceForm::Wide,
                        ecdas::widen,
                    ))
                })
            },
        )?);
        drop(ecdas_ops);
        let hint_ops = Arc::new(hint_ops);
        units.extend(optional_unit(family("HINT"), hint_ops.len(), || {
            Box::new(move || {
                Ok(packed_if(
                    pack,
                    hint::generate_hint_trace_as(&hint_ops.whole(), form),
                ))
            })
        }));
        // BLAKE3 is a table of the proof only when it has ops (`VmAirs`'
        // `include_blake3` from the counts), whatever the build makes of it
        // otherwise. Read off the ops, so a miscounted header is refused below.
        let num_blake3 = blake3_ops.len() + blake3_absorb_ops.len();
        if num_blake3 > 0 {
            let rows = rows_of(num_blake3);
            units.push(Unit::Build {
                family: family("BLAKE3"),
                rows,
                job: Box::new(move || {
                    Ok(packed_if(
                        pack,
                        blake3::generate_blake3_trace(
                            &blake3_ops.whole(),
                            &blake3_absorb_ops.whole(),
                        ),
                    ))
                }),
            });
        }

        // CPU … CPU32.
        units.extend(chunked_units(
            family("CPU"),
            cpu_ops.len(),
            max_rows.cpu,
            skip.cpu,
            tails,
            false,
            |start, end| {
                let ops = Arc::clone(&cpu_ops);
                Box::new(move || {
                    Ok(packed_if(
                        pack,
                        cpu::generate_cpu_trace_as(&ops[start..end], form),
                    ))
                })
            },
        )?);
        drop(cpu_ops);
        let lt = Arc::new(lt);
        units.extend(chunked_units(
            family("LT"),
            lt.segments().len(),
            max_rows.lt,
            skip.lt,
            tails,
            true,
            |start, end| {
                let lt = Arc::clone(&lt);
                Box::new(move || {
                    Ok(packed_if(
                        pack,
                        lt::generate_lt_trace_as(&lt.segments().range(start, end), form),
                    ))
                })
            },
        )?);
        drop(lt);
        units.extend(segmented_units(
            family("SHIFT"),
            shift_ops,
            max_rows.shift,
            skip.shift,
            tails,
            move |ops: &[ShiftOperation]| {
                packed_if(pack, shift::generate_shift_trace_as(ops, form))
            },
        )?);
        units.extend(vec_units(
            family("MEMW"),
            memw_ops,
            max_rows.memw,
            skip.memw,
            tails,
            true,
            move |ops: &[MemwOperation]| packed_if(pack, memw::generate_memw_trace_as(ops, form)),
        )?);
        units.extend(vec_units(
            family("MEMW_A"),
            memw_aligned_ops,
            max_rows.memw_aligned,
            skip.memw_aligned,
            tails,
            true,
            move |ops: &[memw_aligned::AlignedRow]| {
                packed_if(
                    pack,
                    memw_aligned::generate_memw_aligned_trace_as(ops, form),
                )
            },
        )?);
        units.extend(vec_units(
            family("LOAD"),
            load_ops,
            max_rows.load,
            skip.load,
            tails,
            true,
            move |ops: &[LoadOperation]| packed_if(pack, load::generate_load_trace_as(ops, form)),
        )?);
        units.extend(segmented_units(
            family("MUL"),
            mul_ops,
            max_rows.mul,
            0,
            false,
            move |ops: &[(MulOperation, bool)]| {
                packed_if(pack, mul::generate_mul_trace_as(ops, form))
            },
        )?);
        units.extend(segmented_units(
            family("DVRM"),
            dvrm_ops,
            max_rows.dvrm,
            0,
            false,
            move |ops: &[(DvrmOperation, bool)]| {
                packed_if(pack, dvrm::generate_dvrm_trace_as(ops, form))
            },
        )?);
        units.extend(segmented_units(
            family("BRANCH"),
            branch_ops,
            max_rows.branch,
            0,
            false,
            move |ops: &[BranchOperation]| {
                packed_if(pack, branch::generate_branch_trace_as(ops, form))
            },
        )?);
        // PAGE: one a config, from the memory state.
        let page_configs = Arc::new(header.page_configs.clone());
        units.extend((0..page_configs.len()).map(|k| {
            let configs = Arc::clone(&page_configs);
            Unit::Build {
                family: family("PAGE"),
                rows: page::DEFAULT_PAGE_SIZE,
                job: Box::new(move || {
                    Ok(packed_if(
                        pack,
                        page_trace(&configs[k], memory_state, l2g_memory_bookend),
                    ))
                }) as Job<'a>,
            }
        }));
        units.extend(vec_units(
            family("MEMW_R"),
            memw_register_rows,
            max_rows.memw_register,
            skip.memw_register,
            tails,
            false,
            move |ops: &[RegRow]| {
                packed_if(
                    pack,
                    memw_register::generate_memw_register_trace_from_rows_as(ops, form),
                )
            },
        )?);
        units.extend(segmented_units(
            family("EQ"),
            eq_ops,
            max_rows.eq,
            0,
            false,
            move |ops: &[eq::EqOperation]| packed_if(pack, eq::generate_eq_trace_as(ops, form)),
        )?);
        units.extend(segmented_units(
            family("BYTEWISE"),
            bytewise_ops,
            max_rows.bytewise,
            0,
            false,
            move |ops: &[bytewise::BytewiseOperation]| {
                packed_if(pack, bytewise::generate_bytewise_trace_as(ops, form))
            },
        )?);
        units.extend(segmented_units(
            family("STORE"),
            store_ops,
            max_rows.store,
            skip.store,
            tails,
            move |ops: &[store::StoreOperation]| {
                packed_if(pack, store::generate_store_trace_as(ops, form))
            },
        )?);
        units.extend(segmented_units(
            family("CPU32"),
            cpu32_ops,
            max_rows.cpu32,
            0,
            false,
            move |ops: &[cpu32::Cpu32Operation]| {
                packed_if(pack, cpu32::generate_cpu32_trace_as(ops, form))
            },
        )?);

        // The units are the tables the header counted, or the build is refused.
        check_units(&units, &header.table_counts)?;
        Ok(units)
    }
}

/// A family's units from a list held whole ([`chunk_and_generate_skipping`]).
#[allow(clippy::too_many_arguments)]
fn vec_units<'a, T: Send + Sync + 'a>(
    family: usize,
    ops: Vec<T>,
    max: usize,
    skip: usize,
    tails: bool,
    optional: bool,
    generate: impl Fn(&[T]) -> Table + Send + Sync + 'a,
) -> Result<Vec<Unit<'a>>, Error> {
    let len = ops.len();
    let ops = std::sync::Arc::new(ops);
    let generate = std::sync::Arc::new(generate);
    chunked_units(family, len, max, skip, tails, optional, |start, end| {
        let ops = std::sync::Arc::clone(&ops);
        let generate = std::sync::Arc::clone(&generate);
        Box::new(move || Ok(generate(&ops[start..end])))
    })
}

/// A list a family's chunks are read from as segments: in blocks, or
/// delta-coded ([`delta::DeltaStream`]).
trait SegmentList<T>: Send + Sync {
    fn list_len(&self) -> usize;
    fn list_segments(&self) -> Segmented<'_, T>;
}

impl<T: Send + Sync> SegmentList<T> for BlockVec<T> {
    fn list_len(&self) -> usize {
        self.len()
    }

    fn list_segments(&self) -> Segmented<'_, T> {
        self.segments()
    }
}

impl<C: delta::Codec> SegmentList<C::Op> for delta::DeltaStream<C> {
    fn list_len(&self) -> usize {
        self.len()
    }

    fn list_segments(&self) -> Segmented<'_, C::Op> {
        self.segments()
    }
}

/// A family's units from a list read as segments ([`chunk_and_generate_segmented`]
/// over them); an empty list makes no table.
fn segmented_units<'a, T: Clone + Send + Sync + 'a>(
    family: usize,
    ops: impl SegmentList<T> + 'a,
    max: usize,
    skip: usize,
    tails: bool,
    generate: impl Fn(&[T]) -> Table + Send + Sync + 'a,
) -> Result<Vec<Unit<'a>>, Error> {
    let len = ops.list_len();
    let ops = std::sync::Arc::new(ops);
    let generate = std::sync::Arc::new(generate);
    chunked_units(family, len, max, skip, tails, true, |start, end| {
        let ops = std::sync::Arc::clone(&ops);
        let generate = std::sync::Arc::clone(&generate);
        Box::new(move || Ok(generate(&ops.list_segments().range(start, end))))
    })
}

/// KECCAK_RND's units ([`keccak_rnd_chunks`]' tables, or the whole table wide
/// without chunks): a chunk's round ops are derived from the KECCAK ops that
/// reach it as the chunk is built.
fn keccak_rnd_units<'a>(
    keccak_ops: &std::sync::Arc<BlockVec<KeccakOperation>>,
    skip: &StreamSkip,
    pack: bool,
    form: TraceForm,
) -> Result<Vec<Unit<'a>>, Error> {
    let f = family("KECCAK_RND");
    let n = keccak_ops.len();
    let rounds = |ops: &[KeccakOperation]| -> Vec<KeccakRoundOperation> {
        ops.iter()
            .map(|op| KeccakRoundOperation {
                timestamp: op.timestamp,
                input: op.input,
                output: op.output,
            })
            .collect()
    };
    if n == 0 {
        return Ok(Vec::new());
    }
    let rows = skip.keccak_rnd_rows;
    let streamed = skip.keccak_rnd;
    let total = (n * 24).next_power_of_two().max(4);
    if rows == 0 {
        let ops = std::sync::Arc::clone(keccak_ops);
        return Ok(vec![Unit::Build {
            family: f,
            rows: total,
            job: Box::new(move || Ok(keccak_rnd::generate_keccak_rnd_trace(&rounds(&ops.whole())))),
        }]);
    }
    if total <= rows {
        if streamed > 0 {
            return Err(Error::Prover(format!(
                "{streamed} KECCAK_RND chunks were streamed but the table is one of {total} rows"
            )));
        }
        let ops = std::sync::Arc::clone(keccak_ops);
        return Ok(vec![Unit::Build {
            family: f,
            rows: total,
            job: Box::new(move || {
                Ok(packed_if(
                    pack,
                    keccak_rnd::generate_keccak_rnd_rows_as(&rounds(&ops.whole()), 0, total, form),
                ))
            }),
        }]);
    }
    let chunks = total / rows;
    if streamed > chunks {
        return Err(Error::Prover(format!(
            "{streamed} KECCAK_RND chunks were streamed but the run has {chunks}"
        )));
    }
    Ok((0..chunks)
        .map(|c| {
            if c < streamed {
                return Unit::Streamed { family: f };
            }
            let ops = std::sync::Arc::clone(keccak_ops);
            let (first, end) = keccak_rnd_op_range(c, rows, n);
            Unit::Build {
                family: f,
                rows,
                job: Box::new(move || {
                    Ok(packed_if(
                        pack,
                        keccak_rnd::generate_keccak_rnd_rows_as(
                            &rounds(&ops.range(first, end)),
                            c * rows - first * 24,
                            rows,
                            form,
                        ),
                    ))
                }) as Job<'a>,
            }
        })
        .collect())
}

/// The units' tables, family by family, against the header's counts: the same,
/// or an error ([`FinishPlan::emit_all`]'s refusal). HALT and the four fixed
/// tables are one each.
fn check_units(units: &[Unit<'_>], counts: &crate::TableCounts) -> Result<(), Error> {
    let mut made = [0usize; FAMILIES.len()];
    for unit in units {
        let (Unit::Build { family, .. } | Unit::Streamed { family }) = unit;
        if let Some(n) = made.get_mut(*family) {
            *n += 1;
        }
    }
    let of = |name: &str| made[family(name)];
    let units_counts = crate::TableCounts {
        cpu: of("CPU"),
        lt: of("LT"),
        memw: of("MEMW"),
        memw_aligned: of("MEMW_A"),
        load: of("LOAD"),
        mul: of("MUL"),
        dvrm: of("DVRM"),
        shift: of("SHIFT"),
        branch: of("BRANCH"),
        memw_register: of("MEMW_R"),
        eq: of("EQ"),
        bytewise: of("BYTEWISE"),
        store: of("STORE"),
        cpu32: of("CPU32"),
        keccak: of("KECCAK"),
        keccak_rnd: of("KECCAK_RND"),
        ecsm: of("ECSM"),
        ecdas: of("ECDAS"),
        hint: of("HINT"),
        commit: of("COMMIT"),
        blake3: of("BLAKE3"),
    };
    let fixed = ["BITWISE", "DECODE", "KECCAK_RC", "REGISTER", "HALT"];
    if counts_of(&units_counts) != counts_of(counts) || fixed.iter().any(|f| of(f) != 1) {
        return Err(Error::Prover(format!(
            "the streamed finish would build {units_counts:?}, its plan counted {counts:?}"
        )));
    }
    Ok(())
}

impl<'a> FinishPlan<'a> {
    /// Phase 5 streamed: the units ([`Self::units`]) built in waves and handed
    /// to `send` in AIR order, from this thread. A wave is the units in order
    /// up to half the gate's budget of estimated bytes, built in parallel; the
    /// gate holds each table's bytes from before it is built until the packer
    /// places it ([`gate::ByteGate`]), so at most one wave and the tables not
    /// yet placed are in memory. A table is estimated at its family's measured
    /// bytes a row once one of the family has been built, at eight bytes a
    /// cell before. A job's error, or `send`'s, stops the gate and is returned.
    pub(super) fn emit_streamed(
        self,
        gate: &gate::ByteGate,
        send: &mut dyn FnMut(Emitted) -> Result<(), Error>,
    ) -> Result<(), Error> {
        emit_streamed_units(self.units(), gate, send)
    }
}

/// [`FinishPlan::emit_streamed`] over `units`: an error stops the gate, so it
/// reaches every waiter, and is returned.
fn emit_streamed_units(
    units: Result<Vec<Unit<'_>>, Error>,
    gate: &gate::ByteGate,
    send: &mut dyn FnMut(Emitted) -> Result<(), Error>,
) -> Result<(), Error> {
    let result = emit_units(units, gate, send);
    if let Err(e) = &result {
        gate.stop(&format!("the finish stopped: {e:?}"));
    }
    result
}

/// [`emit_streamed_units`]' loop.
fn emit_units(
    units: Result<Vec<Unit<'_>>, Error>,
    gate: &gate::ByteGate,
    send: &mut dyn FnMut(Emitted) -> Result<(), Error>,
) -> Result<(), Error> {
    let wave_bytes = (gate.budget() / 2).max(1);
    // Bytes a row each family's tables took, once one was built.
    let mut per_row: [Option<f64>; FAMILIES.len()] = [None; FAMILIES.len()];
    let estimate = |per_row: &[Option<f64>; FAMILIES.len()], family: usize, rows: usize| {
        let wide = FAMILIES.get(family).map_or(1, |&(_, columns)| columns) * 8;
        let a_row = per_row
            .get(family)
            .copied()
            .flatten()
            .unwrap_or(wide as f64);
        (a_row * rows as f64).ceil() as usize
    };
    let mut units = units?.into_iter().enumerate().peekable();
    let mut next_table = 0usize;
    while units.peek().is_some() {
        // The wave: units in order until one does not fit; its first table
        // waits for its bytes (the lowest not yet placed always goes).
        let mut wave: Vec<WaveSlot<'_>> = Vec::new();
        let mut bytes = 0usize;
        let mut tables = 0usize;
        while let Some((_, unit)) = units.peek() {
            if let Unit::Build { family, rows, .. } = unit {
                let est = estimate(&per_row, *family, *rows);
                if tables == 0 {
                    gate.acquire(next_table, est)?;
                } else if bytes.saturating_add(est) > wave_bytes
                    || !gate.try_acquire(next_table, est)?
                {
                    break;
                }
                bytes = bytes.saturating_add(est);
                tables += 1;
            }
            let Some((position, unit)) = units.next() else {
                break;
            };
            match unit {
                Unit::Streamed { .. } => wave.push((position, None)),
                Unit::Build { family, job, .. } => {
                    wave.push((position, Some((next_table, family, job))));
                    next_table += 1;
                }
            }
        }
        // Built in parallel; no worker waits on the gate or on `send`.
        let (slots, jobs): (Vec<_>, Vec<_>) = wave
            .into_iter()
            .map(|(position, build)| match build {
                Some((table, family, job)) => ((position, Some((table, family))), Some(job)),
                None => ((position, None), None),
            })
            .unzip();
        let jobs: Vec<Job<'_>> = jobs.into_iter().flatten().collect();
        #[cfg(feature = "parallel")]
        let built: Vec<Result<Table, Error>> = jobs.into_par_iter().map(|job| job()).collect();
        #[cfg(not(feature = "parallel"))]
        let built: Vec<Result<Table, Error>> = jobs.into_iter().map(|job| job()).collect();
        let mut built = built.into_iter();
        for (position, build) in slots {
            let table = match build {
                None => None,
                Some((number, family)) => {
                    let table = built.next().ok_or_else(|| {
                        Error::Prover("the streamed finish lost a table of its wave".into())
                    })??;
                    let bytes = table_bytes(&table);
                    gate.resize(number, bytes);
                    if table.main_table.height > 0
                        && let Some(slot) = per_row.get_mut(family)
                    {
                        *slot = Some(bytes as f64 / table.main_table.height as f64);
                    }
                    Some(table)
                }
            };
            send(Emitted { position, table })?;
        }
        build_stamps::mark("p5 wave");
    }
    build_stamps::mark("p5 end");
    Ok(())
}

#[cfg(test)]
#[path = "stream_tests.rs"]
mod stream_tests;
