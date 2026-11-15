//! One Keccak-round group's whole batched argue — GKR trees, ladders and the
//! constraint sumcheck — on the card, at today's KECCAK_RND shape against a
//! bit-sliced two-rounds-a-row layout over the same permutations.
//!
//! ```text
//! LAMBDA_VM_BASE_SPLIT=1 cargo test --release -p lambda-vm-prover --features cuda --lib -- \
//!     --ignored --exact --nocapture tests::keccak_slice_argue_bench::keccak_group_argue_today_against_sliced
//! ```
//!
//! - `today`: four real KECCAK_RND tables (the block's lean form) of 2^16 rows,
//!   from 2,730 random permutations each.
//! - `sliced`: a synthetic two-rounds-a-row table of 2^16 rows (5,460 columns of
//!   digits, the row's 858 lookups at their real shapes, one degree-2
//!   constraint) and a synthetic boundary table of 2^14 rows (3,208 columns:
//!   its 3,200 bits range-checked by 400 byte-to-bits lookups, its four flags
//!   by booleanities; 406 interactions), covering the same permutations.
//! - `fixed`: the format's fixed tables, argued once a block: CHI (2^21 rows,
//!   11 columns and a multiplicity), XOR5 (46,656 rows in 2^16, 6 and one) and
//!   the byte-to-bits table (2^8 rows, 9 and one), each one receiving lookup.
//!
//! The synthetic tables' buses do not balance and nothing is verified: only the
//! shapes are real. Each pass builds the group afresh, puts it on the card as
//! phase B does and argues it once. `KECCAK SLICED` lines; `BATCHED ARGUE SPLIT`
//! follows each argue under `LAMBDA_VM_BASE_SPLIT=1`.

use crypto::fiat_shamir::default_transcript::DefaultTranscript;
use crypto::fiat_shamir::transcript_hash::RpxTranscriptHash;
use multilinear::whir_chain::ArgueFormat;
use stark::constraints::builder::{ConstraintBuilder, ConstraintSet};
use stark::lookup::{
    AirWithBuses, AuxiliaryTraceBuildData, BusInteraction, BusValue, LinearTerm, Multiplicity,
    NullBoundaryConstraintBuilder, Packing,
};
use stark::multilinear_block::reside_group_for_bench;
use stark::multilinear_table::CommittedTable;
use stark::multilinear_table::batched::{ProverFaults, Where, prove_argue};
use stark::traits::AIR;

use crate::constraints::templates::emit_is_bit;
use crate::tables::keccak_rnd::{self, KeccakRndForm, KeccakRoundOperation};
use crate::tables::types::{FE, FEE, GoldilocksExtension, GoldilocksField};

type F = GoldilocksField;
type E = GoldilocksExtension;

/// Booleanity of each listed column (degree 2).
#[derive(Clone)]
struct Bools(Vec<usize>);

impl ConstraintSet<F, E> for Bools {
    fn max_degree(&self) -> usize {
        2
    }

    fn eval<B: ConstraintBuilder<F, E>>(&self, b: &mut B) {
        for (i, &col) in self.0.iter().enumerate() {
            emit_is_bit(b, i, col, None);
        }
    }
}

type BoolAir = AirWithBuses<F, E, NullBoundaryConstraintBuilder, (), Bools>;

fn options() -> stark::proof::options::ProofOptions {
    stark::proof::options::GoldilocksCubicProofOptions::with_params(4, 128, 20).expect("options")
}

fn direct(col: usize) -> BusValue {
    BusValue::Packed {
        start_column: col,
        packing: Packing::Direct,
    }
}

fn sum(cols: &[(usize, i64)]) -> BusValue {
    BusValue::linear(
        cols.iter()
            .map(|&(column, coefficient)| LinearTerm::Column {
                coefficient,
                column,
            })
            .collect(),
    )
}

// The sliced row: ts (4) · round · μ · state_in (1,600) · per round: c (320),
// s (1,600), rc (7).
const SL_TS: usize = 0;
const SL_ROUND: usize = 4;
const SL_MU: usize = 5;
const SL_IN: usize = 6;
const SL_ROUND_WIDTH: usize = 320 + 1600 + 7;
const SL_COLS: usize = SL_IN + 1600 + 2 * SL_ROUND_WIDTH;
const RC_BITS: [usize; 7] = [0, 1, 3, 7, 15, 31, 63];

fn lane(x: usize, y: usize) -> usize {
    x + 5 * y
}

/// Round `k`'s columns: its input state, its parities, its output and its
/// round-constant bits.
fn sl_round(k: usize) -> (usize, usize, usize, usize) {
    let base = SL_IN + 1600 + k * SL_ROUND_WIDTH;
    let input = if k == 0 { SL_IN } else { SL_IN + 1600 + 320 };
    (input, base, base + 320, base + 1920)
}

fn sl_air() -> BoolAir {
    let mu = Multiplicity::Column(SL_MU);
    let mut interactions = Vec::new();
    for k in 0..2 {
        let (input, c, s, rc) = sl_round(k);
        // xor5: three column sums and their parities a lookup.
        for j in 0..107 {
            let mut values = Vec::new();
            let positions: Vec<usize> = (0..3).map(|i| (3 * j + i) % 320).collect();
            for &p in &positions {
                let (x, z) = (p / 64, p % 64);
                let terms: Vec<(usize, i64)> =
                    (0..5).map(|y| (input + lane(x, y) * 64 + z, 1)).collect();
                values.push(sum(&terms));
            }
            for &p in &positions {
                values.push(direct(c + p));
            }
            interactions.push(BusInteraction::sender(100u64, mu.clone(), values));
        }
        // χ + ι: five θ digit-sums, the round-constant bit, five outputs.
        for y in 0..5 {
            for z in 0..64 {
                let mut values = Vec::new();
                for x in 0..5 {
                    let zr = (z + 7 * x + 3 * y) % 64;
                    values.push(sum(&[
                        (input + lane(x, y) * 64 + zr, 1),
                        (c + ((x + 4) % 5) * 64 + zr, 1),
                        (c + ((x + 1) % 5) * 64 + (zr + 63) % 64, 1),
                    ]));
                }
                values.push(match RC_BITS.iter().position(|&b| y == 0 && b == z) {
                    Some(i) => direct(rc + i),
                    None => BusValue::constant(0),
                });
                for x in 0..5 {
                    values.push(direct(s + lane(x, y) * 64 + z));
                }
                interactions.push(BusInteraction::sender(101u64, mu.clone(), values));
            }
        }
        // The round constant: its bytes linear in the seven bits.
        let mut values = vec![sum(&[(SL_ROUND, 1)])];
        values.push(sum(&[(rc, 1), (rc + 1, 2), (rc + 2, 8), (rc + 3, 128)]));
        values.push(sum(&[(rc + 4, 128)]));
        values.push(BusValue::constant(0));
        values.push(sum(&[(rc + 5, 128)]));
        for _ in 0..3 {
            values.push(BusValue::constant(0));
        }
        values.push(sum(&[(rc + 6, 128)]));
        interactions.push(BusInteraction::sender(102u64, mu.clone(), values));
    }
    // The chain: the row's input state in, its output state out.
    let mut recv: Vec<BusValue> = (0..5).map(|i| direct(SL_TS + i)).collect();
    recv.extend((0..1600).map(|i| direct(SL_IN + i)));
    interactions.push(BusInteraction::receiver(103u64, mu.clone(), recv));
    let (_, _, s2, _) = sl_round(1);
    let mut send: Vec<BusValue> = (0..4).map(|i| direct(SL_TS + i)).collect();
    send.push(sum(&[(SL_ROUND, 1)]));
    send.extend((0..1600).map(|i| direct(s2 + i)));
    interactions.push(BusInteraction::sender(103u64, mu, send));
    assert_eq!(interactions.len(), 858);
    AirWithBuses::new(
        SL_COLS,
        AuxiliaryTraceBuildData { interactions },
        &options(),
        1,
        Bools(vec![SL_MU]),
    )
}

// The boundary row: ts (4) · direction · m_a · m_b · m · bits A (1,600) · bits B (1,600).
const SC_TS: usize = 0;
const SC_FLAGS: usize = 4;
const SC_A: usize = 8;
const SC_B: usize = SC_A + 1600;
const SC_COLS: usize = SC_B + 1600;

fn slice_air() -> BoolAir {
    let mut interactions = Vec::new();
    for (bits, ts, m) in [(SC_A, SC_TS, SC_FLAGS + 1), (SC_B, SC_TS + 2, SC_FLAGS + 2)] {
        for round in [0u64, 24] {
            let mut values = vec![direct(ts), direct(ts + 1), BusValue::constant(round)];
            for byte in 0..200 {
                let terms: Vec<(usize, i64)> =
                    (0..8).map(|i| (bits + byte * 8 + i, 1 << i)).collect();
                values.push(sum(&terms));
            }
            interactions.push(if round == 0 {
                BusInteraction::receiver(104u64, Multiplicity::Column(m), values)
            } else {
                BusInteraction::sender(104u64, Multiplicity::Column(m), values)
            });
        }
    }
    for round in [0u64, 24] {
        let mut values: Vec<BusValue> = (0..4).map(|i| direct(SC_TS + i)).collect();
        values.push(BusValue::constant(round));
        values.extend((0..1600).map(|i| sum(&[(SC_A + i, 1), (SC_B + i, 8)])));
        interactions.push(if round == 0 {
            BusInteraction::sender(103u64, Multiplicity::Column(SC_FLAGS + 3), values)
        } else {
            BusInteraction::receiver(103u64, Multiplicity::Column(SC_FLAGS + 3), values)
        });
    }
    // Each byte's bits range-checked against the byte-to-bits table: the byte
    // (linear in its bits) and its eight bits, one lookup a byte.
    for (bits, m) in [(SC_A, SC_FLAGS + 1), (SC_B, SC_FLAGS + 2)] {
        for byte in 0..200 {
            let terms: Vec<(usize, i64)> = (0..8).map(|i| (bits + byte * 8 + i, 1 << i)).collect();
            let mut values = vec![sum(&terms)];
            values.extend((0..8).map(|i| direct(bits + byte * 8 + i)));
            interactions.push(BusInteraction::sender(
                105u64,
                Multiplicity::Column(m),
                values,
            ));
        }
    }
    let bools: Vec<usize> = (SC_FLAGS..SC_A).collect();
    AirWithBuses::new(
        SC_COLS,
        AuxiliaryTraceBuildData { interactions },
        &options(),
        1,
        Bools(bools),
    )
}

/// A fixed table of `width` value columns and a multiplicity, receiving one
/// lookup of all of them on `bus`.
fn fixed_air(width: usize, bus: u64) -> BoolAir {
    let values = (0..width).map(direct).collect();
    AirWithBuses::new(
        width + 1,
        AuxiliaryTraceBuildData {
            interactions: vec![BusInteraction::receiver(
                bus,
                Multiplicity::Column(width),
                values,
            )],
        },
        &options(),
        1,
        Bools(Vec::new()),
    )
}

fn next(seed: &mut u64) -> u64 {
    *seed ^= *seed << 13;
    *seed ^= *seed >> 7;
    *seed ^= *seed << 17;
    *seed
}

/// `width` columns of `2^num_vars` rows, each cell from `cell(column, seed)`.
fn columns(
    width: usize,
    num_vars: usize,
    seed: u64,
    cell: impl Fn(usize, &mut u64) -> u64,
) -> Vec<Vec<FE>> {
    let mut seed = seed | 1;
    (0..width)
        .map(|c| {
            (0..1usize << num_vars)
                .map(|_| FE::from(cell(c, &mut seed)))
                .collect()
        })
        .collect()
}

fn committed<'a>(
    air: &'a dyn AIR<Field = F, FieldExtension = E, PublicInputs = ()>,
    mut cols: Vec<Vec<FE>>,
    num_vars: usize,
) -> CommittedTable<'a, F, E> {
    let layout =
        crate::multilinear_prove::layout_of(air, cols.len(), num_vars).expect("the table lays out");
    CommittedTable::from_layout(layout, |col| core::mem::take(&mut cols[col as usize]))
        .expect("the table builds")
}

fn today_columns(seed: u64, num_vars: usize) -> Vec<Vec<FE>> {
    let mut seed = seed | 1;
    let ops: Vec<KeccakRoundOperation> = (0..(1usize << num_vars) / 24)
        .map(|i| {
            let mut input = [0u64; 25];
            for lane in &mut input {
                *lane = next(&mut seed);
            }
            KeccakRoundOperation {
                timestamp: 8 * i as u64 + 4,
                input,
                output: input,
            }
        })
        .collect();
    keccak_rnd::generate_keccak_rnd_trace(&ops)
        .main_table
        .columns()
}

fn argue(tables: &mut [CommittedTable<'_, F, E>], at: Where) -> f64 {
    if at == Where::Device {
        assert!(reside_group_for_bench(tables), "the card takes the group");
    }
    let ArgueFormat::Batched {
        bin_log_cells,
        pack,
    } = ArgueFormat::BATCHED
    else {
        unreachable!("the production argue is batched")
    };
    let (z, alpha, beta) = (FEE::from(0x2u64), FEE::from(0x3u64), FEE::from(0x5u64));
    let mut transcript = DefaultTranscript::<E, RpxTranscriptHash>::new(&[]);
    let from = std::time::Instant::now();
    prove_argue(
        tables,
        &z,
        &alpha,
        &beta,
        bin_log_cells,
        pack,
        &mut transcript,
        ProverFaults::default(),
        at,
    )
    .expect("the group argues");
    from.elapsed().as_secs_f64()
}

#[test]
#[ignore = "a card bench: box only"]
fn keccak_group_argue_today_against_sliced() {
    const WARM: usize = 2;
    const TIMED: usize = 5;
    let opts = options();
    let today_air = crate::test_utils::create_keccak_rnd_air_for(&opts, KeccakRndForm::Lean);
    let (sl, slice) = (sl_air(), slice_air());
    let (chi, xor5, bytebits) = (fixed_air(11, 101), fixed_air(6, 100), fixed_air(9, 105));
    let built = std::time::Instant::now();
    let today_cols: Vec<Vec<Vec<FE>>> = (0..4).map(|i| today_columns(0x7EA + i, 16)).collect();
    let digit = |_: usize, s: &mut u64| [0u64, 1, 8, 9][(next(s) % 4) as usize];
    let mut sl_cols = columns(SL_COLS, 16, 0x51, digit);
    for row in sl_cols[SL_MU].iter_mut() {
        *row = FE::from(1u64);
    }
    let slice_cols = columns(SC_COLS, 14, 0x5C, |_, s| next(s) & 1);
    let small = |_: usize, s: &mut u64| next(s) % 64;
    let (chi_cols, xor5_cols, bytebits_cols) = (
        columns(12, 21, 0xC1, small),
        columns(7, 16, 0x50, small),
        columns(10, 8, 0xB8, small),
    );
    println!(
        "KECCAK SLICED SETUP: today 4 x KECCAK_RND (lean, {} interactions) x 2^16 · sliced {} columns x 2^16 ({} interactions) + boundary {} columns x 2^14 ({} interactions) · built in {:.1} s",
        today_air.bus_interactions().len(),
        SL_COLS,
        sl.bus_interactions().len(),
        SC_COLS,
        slice.bus_interactions().len(),
        built.elapsed().as_secs_f64()
    );
    let mut times: [Vec<f64>; 3] = [Vec::new(), Vec::new(), Vec::new()];
    for rep in 0..WARM + TIMED {
        let order = if rep % 2 == 0 { [0, 1, 2] } else { [1, 0, 2] };
        for arm in order {
            let name = ["today", "sliced", "fixed"][arm];
            println!(
                "KECCAK SLICED ARGUE arm={name} rep={rep} warm={}",
                rep < WARM
            );
            let secs = if arm == 0 {
                let mut tables: Vec<CommittedTable<'_, F, E>> = today_cols
                    .iter()
                    .map(|c| committed(&today_air, c.clone(), 16))
                    .collect();
                argue(&mut tables, Where::Device)
            } else if arm == 1 {
                let mut tables = vec![
                    committed(&sl, sl_cols.clone(), 16),
                    committed(&slice, slice_cols.clone(), 14),
                ];
                argue(&mut tables, Where::Device)
            } else {
                let mut tables = vec![
                    committed(&chi, chi_cols.clone(), 21),
                    committed(&xor5, xor5_cols.clone(), 16),
                    committed(&bytebits, bytebits_cols.clone(), 8),
                ];
                argue(&mut tables, Where::Device)
            };
            println!(
                "KECCAK SLICED RUN arm={name} rep={rep} warm={} ms={:.3}",
                rep < WARM,
                secs * 1e3
            );
            if rep >= WARM {
                times[arm].push(secs);
            }
        }
    }
    let median = |v: &mut Vec<f64>| {
        v.sort_by(|a, b| a.partial_cmp(b).unwrap());
        v[v.len() / 2]
    };
    let (t, s, f) = (
        median(&mut times[0]),
        median(&mut times[1]),
        median(&mut times[2]),
    );
    println!("KECCAK SLICED MEDIAN arm=today ms={:.3}", t * 1e3);
    println!("KECCAK SLICED MEDIAN arm=sliced ms={:.3}", s * 1e3);
    println!("KECCAK SLICED MEDIAN arm=fixed ms={:.3}", f * 1e3);
    println!(
        "KECCAK SLICED RATIO sliced/today={:.4} (one group's whole argue at equal permutations)",
        s / t
    );
}

/// The bench's tables lay out and argue, on the host at a small size: today's
/// four KECCAK_RND tables of 2^8 rows, the sliced table of 2^8 and the boundary
/// of 2^6 — every AIR, interaction and column the card bench uses.
#[test]
fn the_bench_tables_argue_at_a_small_size() {
    let opts = options();
    let today_air = crate::test_utils::create_keccak_rnd_air_for(&opts, KeccakRndForm::Lean);
    let (sl, slice) = (sl_air(), slice_air());
    let mut today: Vec<CommittedTable<'_, F, E>> = (0..4)
        .map(|i| committed(&today_air, today_columns(0x7EA + i, 8), 8))
        .collect();
    argue(&mut today, Where::Host);
    let digit = |_: usize, s: &mut u64| [0u64, 1, 8, 9][(next(s) % 4) as usize];
    let mut sl_cols = columns(SL_COLS, 8, 0x51, digit);
    for row in sl_cols[SL_MU].iter_mut() {
        *row = FE::from(1u64);
    }
    let mut sliced = vec![
        committed(&sl, sl_cols, 8),
        committed(&slice, columns(SC_COLS, 6, 0x5C, |_, s| next(s) & 1), 6),
    ];
    argue(&mut sliced, Where::Host);
    assert_eq!(sl.bus_interactions().len(), 858);
    assert_eq!(slice.bus_interactions().len(), 406);
    let bytebits = fixed_air(9, 105);
    let mut fixed = vec![committed(
        &bytebits,
        columns(10, 8, 0xB8, |_, s| next(s) % 64),
        8,
    )];
    argue(&mut fixed, Where::Host);
}
