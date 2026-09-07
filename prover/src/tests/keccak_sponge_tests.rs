//! KECCAK_SPONGE chip unit tests.
//!
//! The main test is a full sender ↔ collector **multiset equality**: the
//! BITWISE lookups tallied by `collect_bitwise_from_keccak_sponge` (which fill
//! the BITWISE table's multiplicities) must equal, as a multiset of concrete
//! `(bus, tuple)` values, exactly what the KECCAK_SPONGE chip and the
//! KECCAK_RND rows it drives *send* on the ByteAlu/AreBytes buses — evaluated
//! off the real generated traces, not re-derived from the op structure. Any
//! drift between `bus_interactions()` and the collector leaves those buses
//! unbalanced and every sponge proof invalid.

use std::collections::HashMap;

use stark::lookup::{BusValue, LinearTerm, Multiplicity, Packing};
use stark::trace::TraceTable;

use crate::tables::bitwise::{BitwiseOperation, BitwiseOperationType};
use crate::tables::keccak_rnd::{self, KeccakRoundOperation};
use crate::tables::keccak_sponge::{
    self, KeccakSpongeOperation, RATE_BYTES, generate_keccak_sponge_trace,
};
use crate::tables::trace_builder::collect_bitwise_from_keccak_sponge;
use crate::tables::types::{BusId, FE, GoldilocksExtension, GoldilocksField, alu_op};

/// Deterministic SplitMix64.
struct SplitMix64(u64);
impl SplitMix64 {
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
}

/// Build one synthetic absorb call of `n_blocks` blocks with seeded
/// pseudo-random state and data, using the executor's permutation (the same
/// construction as `collect_keccak_sponge_ops`).
fn synthetic_call(n_blocks: u64, timestamp: u64, seed: u64) -> Vec<KeccakSpongeOperation> {
    let mut rng = SplitMix64(seed);
    let state_addr = 0x0001_2340u64;
    let data_addr = 0x0002_0000u64;

    let mut state: [u64; 25] = core::array::from_fn(|i| rng.next_u64() ^ (i as u64));
    let mut ops = Vec::with_capacity(n_blocks as usize);
    for k in 0..n_blocks {
        let mut block = [0u8; RATE_BYTES];
        for chunk in block.chunks_mut(8) {
            chunk.copy_from_slice(&rng.next_u64().to_le_bytes());
        }
        let state_in = state;
        for (j, lane) in state.iter_mut().take(17).enumerate() {
            let mut m = 0u64;
            for b in 0..8 {
                m |= (block[j * 8 + b] as u64) << (b * 8);
            }
            *lane ^= m;
        }
        executor::vm::instruction::execution::keccak_f1600(&mut state);
        ops.push(KeccakSpongeOperation {
            timestamp,
            seq: k,
            n_blocks,
            state_addr,
            block_addr: data_addr + k * RATE_BYTES as u64,
            state_in,
            block,
            state_out: state,
            first: k == 0,
            last: k == n_blocks - 1,
        });
    }
    ops
}

/// Evaluate one bus element on a trace row.
fn eval_bus_value(
    v: &BusValue,
    trace: &TraceTable<GoldilocksField, GoldilocksExtension>,
    row: usize,
) -> FE {
    match v {
        BusValue::Packed {
            start_column,
            packing: Packing::Direct,
        } => *trace.get_main(row, *start_column),
        BusValue::Packed { .. } => panic!("unexpected non-Direct packing in a lookup tuple"),
        BusValue::Linear(terms) => terms.iter().fold(FE::zero(), |acc, t| {
            acc + match t {
                LinearTerm::Column {
                    coefficient,
                    column,
                } => {
                    let cell = *trace.get_main(row, *column);
                    if *coefficient >= 0 {
                        cell * FE::from(*coefficient as u64)
                    } else {
                        -(cell * FE::from(coefficient.unsigned_abs()))
                    }
                }
                LinearTerm::ColumnUnsigned {
                    coefficient,
                    column,
                } => *trace.get_main(row, *column) * FE::from(*coefficient),
                LinearTerm::Constant(c) => {
                    if *c >= 0 {
                        FE::from(*c as u64)
                    } else {
                        -FE::from(c.unsigned_abs())
                    }
                }
            }
        }),
    }
}

type LookupKey = (u64, Vec<u64>);

/// Tally every ByteAlu/AreBytes SEND of `interactions` evaluated over the
/// real rows of `trace` into `sends` (key = (bus_id, canonical tuple)).
fn tally_lookup_sends(
    interactions: &[stark::lookup::BusInteraction],
    trace: &TraceTable<GoldilocksField, GoldilocksExtension>,
    sends: &mut HashMap<LookupKey, i64>,
) {
    let lookup_buses = [BusId::ByteAlu as u64, BusId::AreBytes as u64];
    for interaction in interactions {
        if !lookup_buses.contains(&interaction.bus_id) {
            continue;
        }
        assert!(interaction.is_sender, "lookup interactions are sends");
        let mult_col = match interaction.multiplicity {
            Multiplicity::Column(c) => c,
            _ => panic!("sponge/rnd lookup sends use Multiplicity::Column"),
        };
        for row in 0..trace.num_rows() {
            let mult = trace.get_main(row, mult_col).canonical_u64();
            if mult == 0 {
                continue;
            }
            let tuple: Vec<u64> = interaction
                .values
                .iter()
                .map(|v| eval_bus_value(v, trace, row).canonical_u64())
                .collect();
            *sends.entry((interaction.bus_id, tuple)).or_default() += mult as i64;
        }
    }
}

/// Canonicalize a collected `BitwiseOperation` into the same key space as the
/// evaluated sends.
fn collected_key(op: &BitwiseOperation) -> LookupKey {
    let (x, y) = (op.x as u64, op.y as u64);
    match op.lookup_type {
        BitwiseOperationType::ByteAluXor => {
            (BusId::ByteAlu as u64, vec![alu_op::XOR as u64, x, y, x ^ y])
        }
        BitwiseOperationType::ByteAluAnd => {
            (BusId::ByteAlu as u64, vec![alu_op::AND as u64, x, y, x & y])
        }
        BitwiseOperationType::AreBytes => (BusId::AreBytes as u64, vec![x, y]),
        other => panic!("KECCAK_SPONGE collector emitted unexpected lookup type {other:?}"),
    }
}

/// The multiset of BITWISE lookups the collector tallies must equal the
/// multiset the KECCAK_SPONGE chip + its KECCAK_RND rows actually send,
/// evaluated off the generated traces. Exercises multi-block calls (bookend
/// rows AND interior rows) plus an n = 1 call (a row that is both first and
/// last) sharing the table with it.
#[test]
fn sponge_bitwise_multiset_matches_chip_sends() {
    let mut ops = synthetic_call(3, 4, 0x5EED_0001);
    ops.extend(synthetic_call(1, 8, 0x5EED_0002));

    // The KECCAK_RND rows this sponge workload drives: one permutation per
    // block, input = the absorbed state (mirrors `gen_keccak_rnd`).
    let rnd_ops: Vec<KeccakRoundOperation> = ops
        .iter()
        .map(|op| {
            let mut absorbed = op.state_in;
            for (j, lane) in absorbed.iter_mut().take(17).enumerate() {
                let mut m = 0u64;
                for b in 0..8 {
                    m |= (op.block[j * 8 + b] as u64) << (b * 8);
                }
                *lane ^= m;
            }
            KeccakRoundOperation {
                timestamp: op.timestamp,
                seq: op.seq,
                input: absorbed,
                output: op.state_out,
            }
        })
        .collect();

    let sponge_trace = generate_keccak_sponge_trace(&ops);
    let rnd_trace = keccak_rnd::generate_keccak_rnd_trace(&rnd_ops);

    let mut sends: HashMap<LookupKey, i64> = HashMap::new();
    tally_lookup_sends(
        &keccak_sponge::bus_interactions(),
        &sponge_trace,
        &mut sends,
    );
    tally_lookup_sends(&keccak_rnd::bus_interactions(), &rnd_trace, &mut sends);

    let mut collected: HashMap<LookupKey, i64> = HashMap::new();
    for op in collect_bitwise_from_keccak_sponge(&ops) {
        *collected.entry(collected_key(&op)).or_default() += 1;
    }

    // Compare as full multisets, reporting the first divergence legibly.
    for (key, &count) in &sends {
        assert_eq!(
            collected.get(key).copied().unwrap_or(0),
            count,
            "collector under/over-tallies chip send {key:?}"
        );
    }
    for (key, &count) in &collected {
        assert_eq!(
            sends.get(key).copied().unwrap_or(0),
            count,
            "collector tallies a lookup the chip never sends: {key:?}"
        );
    }
}

/// The sponge chip must not send IS_HALF (it uses linear low-limb addressing,
/// not the KECCAK core chip's DWordHL pointer apparatus), and the collector
/// must mirror that.
#[test]
fn sponge_sends_no_is_half() {
    let is_half = BusId::IsHalfword as u64;
    assert!(
        keccak_sponge::bus_interactions()
            .iter()
            .all(|i| i.bus_id != is_half),
        "sponge chip unexpectedly sends IS_HALF"
    );
    let ops = synthetic_call(2, 4, 0x5EED_0003);
    assert!(
        collect_bitwise_from_keccak_sponge(&ops)
            .iter()
            .all(|op| op.lookup_type != BitwiseOperationType::IsHalf),
        "sponge collector unexpectedly tallies IS_HALF"
    );
}
