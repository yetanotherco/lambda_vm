//! LFM trace generation: instruction column group (preprocessed, leading)
//! plus value columns from the executor's records. Aux/LogUp columns are
//! entirely framework-built; heights equal each group's padded height, so the
//! prover's preprocessed-subset recommit matches the registry root exactly.

use stark::trace::TraceTable;

use crate::tables::types::{FE, GoldilocksExtension, GoldilocksField};

use crate::tables::{bitwise, keccak_rc, keccak_rnd};

use super::chips::{balu, bitdec, const_, hash, hint, keccak, lanes, public, select, xalu};
use super::compiler::{ColumnGroup, LfmProgram};
use super::executor::LfmRecords;
use super::hash::{LfmHasher, TestPermutation};
use super::instr::{HashMode, Instr};
use super::keccak_adapter::{self, KeccakAdapterOperation};
use super::layout;

type F = GoldilocksField;
type E = GoldilocksExtension;

pub struct LfmTraces {
    pub const_: TraceTable<F, E>,
    pub balu: TraceTable<F, E>,
    pub xalu: TraceTable<F, E>,
    pub select: TraceTable<F, E>,
    pub bitdec: TraceTable<F, E>,
    pub hash: TraceTable<F, E>,
    pub keccak: TraceTable<F, E>,
    pub lanes: TraceTable<F, E>,
    pub hint: TraceTable<F, E>,
    pub public: TraceTable<F, E>,
    pub range: TraceTable<F, E>,
    /// The three production keccak-family tables, proved unchanged. They carry
    /// no LFM instruction column group: `KECCAK_RND` has no preprocessed
    /// columns at all, and the other two have fixed, program-independent ones.
    pub keccak_rnd: TraceTable<F, E>,
    pub keccak_rc: TraceTable<F, E>,
    pub bitwise: TraceTable<F, E>,
}

/// The `LFM_RANGE` fixed table's column group (program-independent).
pub fn range_group() -> ColumnGroup {
    ColumnGroup {
        width: layout::range::PREP_WIDTH,
        real_rows: layout::range::NUM_ROWS,
        padded_rows: layout::range::NUM_ROWS,
        data: (0..layout::range::NUM_ROWS as u64).map(FE::from).collect(),
    }
}

/// Builds one chip's trace: copy the (already padded) group into the leading
/// columns, then let `fill` write the value columns of each real row.
fn chip_trace(
    group: &ColumnGroup,
    num_columns: usize,
    mut fill: impl FnMut(usize, &mut [FE]),
) -> TraceTable<F, E> {
    let rows = group.padded_rows;
    let mut data = vec![FE::zero(); rows * num_columns];
    for row in 0..rows {
        data[row * num_columns..row * num_columns + group.width]
            .copy_from_slice(&group.data[row * group.width..(row + 1) * group.width]);
    }
    for row in 0..group.real_rows {
        fill(row, &mut data[row * num_columns..(row + 1) * num_columns]);
    }
    TraceTable::new_main(data, num_columns, 1)
}

pub fn build_traces(program: &LfmProgram, records: &LfmRecords) -> LfmTraces {
    let g = &program.groups;

    let hash_modes: Vec<HashMode> = program
        .instrs
        .iter()
        .filter_map(|i| match i {
            Instr::Hash { mode, .. } => Some(*mode),
            _ => None,
        })
        .collect();
    let iv = TestPermutation.compress_iv();

    // The keccak family's traces are driven by the executor's records; the tag
    // is the row ordinal, exactly as the compiler emitted it into the
    // preprocessed group (one rule, `layout::keccak::tag_for_row`, two callers).
    // The family sees `perm_in` — post-XOR on absorb rows — not the state as
    // read from memory.
    let keccak_ops: Vec<KeccakAdapterOperation> = records
        .keccak
        .iter()
        .enumerate()
        .map(|(row, r)| KeccakAdapterOperation {
            tag: layout::keccak::tag_for_row(row),
            input: r.perm_in,
        })
        .collect();

    let keccak_rnd_trace =
        keccak_rnd::generate_keccak_rnd_trace(&keccak_adapter::round_operations(&keccak_ops));

    let mut keccak_rc_trace = keccak_rc::generate_keccak_rc_trace();
    keccak_rc::update_multiplicities(&mut keccak_rc_trace, keccak_ops.len());

    let mut histogram = bitwise::BitwiseHistogram::new();
    histogram.add_ops(&keccak_adapter::bitwise_ops_for(&keccak_ops));
    // Absorb rows additionally send one BYTE_ALU[XOR] lookup per rate byte.
    histogram.add_ops(&keccak_adapter::absorb_bitwise_ops(&records.keccak));
    let mut bitwise_trace = bitwise::generate_bitwise_trace();
    histogram.fill_multiplicities(&mut bitwise_trace);

    LfmTraces {
        const_: chip_trace(&g.const_, const_::cols::NUM_COLUMNS, |_, _| {}),
        balu: chip_trace(&g.balu, balu::cols::NUM_COLUMNS, |row, out| {
            let r = &records.balu[row];
            out[balu::cols::A] = r.a;
            out[balu::cols::B] = r.b;
            out[balu::cols::C] = r.c;
            out[balu::cols::OUT] = r.out;
        }),
        xalu: chip_trace(&g.xalu, xalu::cols::NUM_COLUMNS, |row, out| {
            let r = &records.xalu[row];
            out[xalu::cols::A0..xalu::cols::A0 + 3].copy_from_slice(&r.a);
            out[xalu::cols::B0..xalu::cols::B0 + 3].copy_from_slice(&r.b);
            out[xalu::cols::C0..xalu::cols::C0 + 3].copy_from_slice(&r.c);
            out[xalu::cols::OUT0..xalu::cols::OUT0 + 3].copy_from_slice(&r.out);
        }),
        select: chip_trace(&g.select, select::cols::NUM_COLUMNS, |row, out| {
            let r = &records.select[row];
            out[select::cols::BIT] = r.bit;
            out[select::cols::INL0..select::cols::INL0 + 4].copy_from_slice(&r.in_l);
            out[select::cols::INR0..select::cols::INR0 + 4].copy_from_slice(&r.in_r);
            out[select::cols::OUTL0..select::cols::OUTL0 + 4].copy_from_slice(&r.out_l);
            out[select::cols::OUTR0..select::cols::OUTR0 + 4].copy_from_slice(&r.out_r);
        }),
        bitdec: chip_trace(&g.bitdec, bitdec::cols::NUM_COLUMNS, |row, out| {
            let r = &records.bitdec[row];
            out[bitdec::cols::BITS0..bitdec::cols::BITS0 + 64].copy_from_slice(&r.bits);
            out[bitdec::cols::Z] = r.z;
            out[bitdec::cols::GINV] = r.ginv;
        }),
        hash: chip_trace(&g.hash, hash::cols::NUM_COLUMNS, |row, out| {
            let r = &records.hash[row];
            out[hash::cols::IN0..hash::cols::IN0 + 12].copy_from_slice(&r.ins);
            for k in 0..4 {
                // S_i = MODE_P·IN_i + MODE_C·IV_i, materialized.
                out[hash::cols::S8 + k] = match hash_modes[row] {
                    HashMode::Permute => r.ins[8 + k],
                    HashMode::Compress => iv[k],
                };
            }
            out[hash::cols::OUT0..hash::cols::OUT0 + 12].copy_from_slice(&r.outs);
        }),
        keccak: chip_trace(&g.keccak, keccak::cols::NUM_COLUMNS, |row, out| {
            let r = &records.keccak[row];
            for lane in 0..25 {
                for b in 0..8 {
                    let byte = |v: u64| FE::from(u64::from((v >> (8 * b)) as u8));
                    out[keccak::cols::state_byte(lane, b)] = byte(r.state[lane]);
                    out[keccak::cols::perm_in_byte(lane, b)] = byte(r.perm_in[lane]);
                    out[keccak::cols::out_byte(lane, b)] = byte(r.output[lane]);
                }
            }
            for (k, &v) in r.block.iter().enumerate() {
                out[keccak::cols::BLOCK + k] = FE::from(u64::from(v));
            }
        }),
        lanes: chip_trace(&g.lanes, lanes::cols::NUM_COLUMNS, |row, out| {
            out[lanes::cols::V0..lanes::cols::V0 + 4].copy_from_slice(&records.lanes[row]);
        }),
        hint: chip_trace(&g.hint, hint::cols::NUM_COLUMNS, |row, out| {
            out[hint::cols::V0..hint::cols::V0 + 4].copy_from_slice(&records.hint[row]);
        }),
        public: chip_trace(&g.public, public::cols::NUM_COLUMNS, |row, out| {
            out[public::cols::V0..public::cols::V0 + 4].copy_from_slice(&records.public[row]);
        }),
        range: chip_trace(
            &range_group(),
            super::chips::range::cols::NUM_COLUMNS,
            |_, _| {},
        ),
        keccak_rnd: keccak_rnd_trace,
        keccak_rc: keccak_rc_trace,
        bitwise: bitwise_trace,
    }
}
