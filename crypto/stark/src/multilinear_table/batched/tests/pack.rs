//! The GKR slot packing (`GkrPack::Chunks`):
//! the chunking rule, the combined claim, round trips of cut tables, the
//! tampers it must refuse, and that neither format's verifier takes the
//! other's proof.

use super::*;
use multilinear::mle::Mle;

use crate::lookup::{BusInteraction, Multiplicity, Packing};

fn packed(bin_log_cells: u8, tree_log_cells: u8) -> ChainConfig {
    config(ArgueFormat::Batched {
        bin_log_cells,
        pack: GkrPack::Chunks { tree_log_cells },
    })
}

/// The batched format at `bin_log_cells` with one tree a table.
fn unpacked(bin_log_cells: u8) -> ChainConfig {
    config(ArgueFormat::Batched {
        bin_log_cells,
        pack: GkrPack::Off,
    })
}

fn shape_of(interactions: usize, num_vars: usize) -> ArgueShape {
    ArgueShape {
        num_vars,
        input_vars: multilinear::logup::input_layer_vars(interactions, num_vars),
        degree: 2,
        shifted: false,
        interactions,
    }
}

fn cut(interactions: usize, num_vars: usize, tree_log_cells: u8) -> (usize, usize) {
    let chunks = chunking(
        &shape_of(interactions, num_vars),
        GkrPack::Chunks { tree_log_cells },
    );
    (chunks.slot_vars, chunks.count)
}

// ─── the rule ────────────────────────────────────────────────────────────

/// ★ Over every interaction count to 2,100, every height to 24 and a spread of
/// prices: the chunks cover the interactions with less than a chunk to spare,
/// they and the all-padding chunks after them fill today's slots exactly (so
/// the combined claim has today's length), a cut saves at its own price, and
/// `Off` is today's one tree.
#[test]
fn the_chunking_rule_keeps_its_invariants() {
    for m in 1..=2100usize {
        for n in [0usize, 1, 3, 7, 16, 20, 21, 24] {
            let shape = shape_of(m, n);
            let whole = Chunking::whole(&shape);
            assert_eq!(chunking(&shape, GkrPack::Off), whole);
            assert_eq!(whole.count, 1);
            for tau in [0u8, 2, 8, 16, 21, 22, 23, 30, 40] {
                let c = chunking(
                    &shape,
                    GkrPack::Chunks {
                        tree_log_cells: tau,
                    },
                );
                let b = c.slot_vars;
                assert!(c.count << b >= m, "m={m} n={n} τ={tau}: the chunks cover");
                assert!(
                    (c.count - 1) << b < m,
                    "m={m} n={n} τ={tau}: no empty chunk"
                );
                assert_eq!(
                    b + c.combine_vars(),
                    whole.slot_vars,
                    "m={m} n={n} τ={tau}: today's slots"
                );
                if c.count > 1 {
                    let cost = |c: &Chunking| {
                        ((c.count as u128) << (n + c.slot_vars))
                            + (c.count as u128 - 1) * (1u128 << tau)
                    };
                    assert!(cost(&c) < cost(&whole), "m={m} n={n} τ={tau}: a cut saves");
                    assert!(c.count >= 3, "m={m} n={n} τ={tau}: two chunks never save");
                }
            }
        }
    }
}

/// The production setting's cuts on the base's tables (their measured
/// shapes): τ = 22 cuts CPU, MEMW_A and SHIFT in three and KECCAK_RND in five,
/// and nothing whose padding is cheaper than the trees; τ = 23 cuts only
/// KECCAK_RND, in three (five ties with three there, and ties go to fewer
/// trees).
#[test]
fn the_production_settings_cut_the_named_tables() {
    let tables = [
        ("CPU", 20usize, 21usize, (3usize, 3usize), None),
        ("KECCAK_RND", 1031, 16, (8, 5), Some((9, 3))),
        ("MEMW_A", 20, 21, (3, 3), None),
        ("SHIFT", 18, 21, (3, 3), None),
        ("MEMW_R", 7, 21, (3, 1), None),
        ("LT", 9, 20, (4, 1), None),
        ("STORE", 10, 21, (4, 1), None),
        ("LOAD", 5, 21, (3, 1), None),
        ("MEMW", 26, 21, (5, 1), None),
        ("ECDAS", 388, 17, (9, 1), None),
        ("KECCAK", 134, 17, (8, 1), None),
        ("PAGE", 3, 18, (2, 1), None),
    ];
    for (name, m, n, at22, at23) in tables {
        assert_eq!(cut(m, n, 22), at22, "{name} at τ = 22");
        let whole = (shape_of(m, n).input_vars - n, 1);
        assert_eq!(cut(m, n, 23), at23.unwrap_or(whole), "{name} at τ = 23");
    }
}

/// ★ At `Off` the prover's trees are today's: one a table, each over the
/// table's own interaction list (borrowed, the same slice), in table order —
/// so the device path builds, queues and steps exactly what it did before.
#[test]
fn the_off_layout_is_one_borrowed_tree_a_table() {
    let airs = airs();
    let fans = fan_airs();
    let fan_cols = fan_cols();
    let cfg = unpacked(27);
    let committed = commit(packed_tables(&airs, &fans, &fan_cols), &cfg);
    let plan = plan_of(&committed, &cfg);
    assert_eq!(plan.trees(), committed.tables().len());
    let z = ExtE::from(7u64);
    let alpha = ExtE::from(11u64);
    for (t, table) in committed.tables().iter().enumerate() {
        assert_eq!(plan.first_tree(t), t);
        assert_eq!(table_of_tree(&plan, t), t);
        let statement = table.statement();
        let interactions = multilinear_logup::interactions(
            statement.interactions,
            statement.slot_of.len(),
            &z,
            &alpha,
            |col| slot(statement.slot_of, col),
        )
        .expect("interactions");
        let lists = chunk_lists(&interactions, &plan.chunks[t]);
        assert_eq!(lists.len(), 1, "table {t}: one tree");
        match &lists[0] {
            std::borrow::Cow::Borrowed(list) => assert!(
                std::ptr::eq(list.as_ptr(), interactions.as_ptr())
                    && list.len() == interactions.len(),
                "table {t}: the table's own list"
            ),
            std::borrow::Cow::Owned(_) => panic!("table {t}: a copied list at Off"),
        }
    }
}

/// At a price no cut can beat, the packed plan is today's plan.
#[test]
fn a_prohibitive_price_plans_todays_bins() {
    let shapes: Vec<ArgueShape> = [(20, 21), (7, 21), (1031, 16), (9, 20), (3, 18), (5, 21)]
        .iter()
        .map(|&(m, n)| shape_of(m, n))
        .collect();
    for cap in [26u8, 27, 20, 0] {
        assert_eq!(
            argue_plan_packed(&shapes, cap, GkrPack::Chunks { tree_log_cells: 40 }),
            argue_plan(&shapes, cap),
            "cap {cap}"
        );
    }
    // And a cut table's unit is its chunks' cells: CPU in three chunks of
    // 2^24 no longer sits alone under 2^26, so a MEMW_R joins it.
    let plan = argue_plan_packed(&shapes, 26, GkrPack::Chunks { tree_log_cells: 22 });
    assert_eq!(
        plan.chunks[0],
        Chunking {
            slot_vars: 3,
            count: 3
        }
    );
    assert!(plan.bins.contains(&vec![0, 1]), "{:?}", plan.bins);
}

// ─── the combined claim ──────────────────────────────────────────────────

fn next(seed: &mut u64) -> u64 {
    *seed ^= *seed << 13;
    *seed ^= *seed >> 7;
    *seed ^= *seed << 17;
    *seed
}

fn ext(seed: &mut u64) -> ExtE {
    ExtE::new([
        FE::from(next(seed)),
        FE::from(next(seed)),
        FE::from(next(seed)),
    ])
}

/// ★ The chunks' claims at one point, combined at `s'`, are the padded
/// layer's multilinear extension at `s' ‖ P`: today's claim. Every count of
/// chunks, so the all-padding chunks' `q ≡ 1` term is exercised too.
#[test]
fn combined_chunk_claims_are_the_padded_layers_extension() {
    let mut seed = 0x9AC4_u64;
    for (m, n, b) in [
        (5usize, 2usize, 0usize),
        (7, 1, 0),
        (9, 2, 2),
        (11, 3, 2),
        (20, 2, 3),
        (6, 0, 1),
    ] {
        let slots = m.next_power_of_two();
        let rows = 1usize << n;
        // The padded layer: interaction-major, `0/1` past `m`.
        let (mut p, mut q) = (Vec::new(), Vec::new());
        for i in 0..slots {
            for _ in 0..rows {
                if i < m {
                    p.push(ext(&mut seed));
                    q.push(ext(&mut seed));
                } else {
                    p.push(ExtE::zero());
                    q.push(ExtE::one());
                }
            }
        }
        let chunk = (1usize << b) * rows;
        let count = m.div_ceil(1 << b);
        let point: Vec<ExtE> = (0..b + n).map(|_| ext(&mut seed)).collect();
        let claims: Vec<GkrClaim<Ext>> = (0..count)
            .map(|j| GkrClaim {
                point: point.clone(),
                p: Mle::new(p[j * chunk..(j + 1) * chunk].to_vec())
                    .unwrap()
                    .evaluate(&point)
                    .unwrap(),
                q: Mle::new(q[j * chunk..(j + 1) * chunk].to_vec())
                    .unwrap()
                    .evaluate(&point)
                    .unwrap(),
            })
            .collect();
        let ell = count.next_power_of_two().trailing_zeros() as usize;
        assert_eq!(b + ell, slots.trailing_zeros() as usize, "m={m} b={b}");
        let combine: Vec<ExtE> = (0..ell).map(|_| ext(&mut seed)).collect();
        let claim = combine_chunk_claims(&claims, &combine).expect("combines");
        let at: Vec<ExtE> = combine.iter().chain(&point).cloned().collect();
        assert_eq!(claim.point, at, "m={m} b={b}: the point");
        assert_eq!(
            claim.p,
            Mle::new(p.clone()).unwrap().evaluate(&at).unwrap(),
            "m={m} b={b}: p̃"
        );
        assert_eq!(
            claim.q,
            Mle::new(q.clone()).unwrap().evaluate(&at).unwrap(),
            "m={m} b={b}: q̃"
        );
        // Chunks at two points do not combine.
        let mut apart = claims.clone();
        apart[count - 1].point[0] += ExtE::one();
        assert!(
            matches!(
                combine_chunk_claims(&apart, &combine),
                Err(MlError::ArgueShapeMismatch { part: "chunks" })
            ),
            "m={m} b={b}: chunks at two points"
        );
    }
}

// ─── round trips ─────────────────────────────────────────────────────────

/// A table of `m` interactions over one column `x` that balances by itself:
/// pair `k` sends `x` on bus `100 + k` and receives it back, and an odd last
/// one sends `x` on bus 200 with the zero column for multiplicity. Every
/// interaction is a different fraction, so chunks in the wrong order are
/// another statement.
fn fan_air(m: usize) -> Air<EmptyConstraints> {
    let options = ProofOptions::default_test_options();
    let interactions: Vec<BusInteraction> = (0..m)
        .map(|i| {
            let x = Packing::Direct.columns(&[0]);
            if i + 1 == m && m % 2 == 1 {
                BusInteraction::sender(200u64, Multiplicity::Column(1), x)
            } else if i % 2 == 0 {
                BusInteraction::sender(100 + (i / 2) as u64, Multiplicity::One, x)
            } else {
                BusInteraction::receiver(100 + (i / 2) as u64, Multiplicity::One, x)
            }
        })
        .collect();
    AirWithBuses::new(
        2,
        AuxiliaryTraceBuildData { interactions },
        &options,
        1,
        EmptyConstraints,
    )
}

/// `x` distinct on every row, the zero column beside it.
fn fan_columns(num_vars: usize, salt: u64) -> Vec<Vec<FE>> {
    let rows = 1u64 << num_vars;
    vec![
        (0..rows)
            .map(|r| FE::from(salt * 1000 + 3 * r + 1))
            .collect(),
        vec![FE::zero(); rows as usize],
    ]
}

/// The fan tables of a run: `(interactions, height)`.
const FANS: [(usize, usize); 4] = [(5, 3), (7, 5), (9, 3), (11, 3)];

fn fan_airs() -> Vec<Air<EmptyConstraints>> {
    FANS.iter().map(|&(m, _)| fan_air(m)).collect()
}

/// CPU, ADD and MUL over `columns(3, 1)`, then the fans.
fn packed_tables<'a>(
    airs: &'a Airs,
    fans: &'a [Air<EmptyConstraints>],
    fan_cols: &[Vec<Vec<FE>>],
) -> Vec<CommittedTable<'a, Fp, Ext>> {
    let cols = columns(3, 1);
    let mut all = tables(airs, &cols, false);
    for (air, cols) in fans.iter().zip(fan_cols) {
        all.push(table(air, cols));
    }
    all
}

fn fan_cols() -> Vec<Vec<Vec<FE>>> {
    FANS.iter()
        .enumerate()
        .map(|(i, &(_, n))| fan_columns(n, i as u64 + 1))
        .collect()
}

fn plan_of(committed: &CommittedTables<'_, Fp, Ext, KeccakWhir>, cfg: &ChainConfig) -> ArguePlan {
    let ArgueFormat::Batched {
        bin_log_cells,
        pack,
    } = cfg.format.argue
    else {
        unreachable!("a batched config")
    };
    let shapes: Vec<ArgueShape> = committed
        .tables()
        .iter()
        .map(|t| ArgueShape::of(&t.statement()))
        .collect();
    argue_plan_packed(&shapes, bin_log_cells, pack)
}

/// ★ Cut tables round trip — in one bin, in small bins, every table alone —
/// with one bus output a tree. At τ = 2 every fan is cut: FAN(5) and FAN(9)
/// in chunks of two whose last holds one interaction (padded to its slots),
/// FAN(7) in seven chunks of one, FAN(11) in three of four; each with
/// all-padding chunks after its last.
#[test]
fn cut_tables_round_trip_across_bins() {
    let airs = airs();
    let fans = fan_airs();
    let fan_cols = fan_cols();
    for cap in [27u8, 6, 4, 0] {
        let cfg = packed(cap, 2);
        let committed = commit(packed_tables(&airs, &fans, &fan_cols), &cfg);
        let plan = plan_of(&committed, &cfg);
        let cuts: Vec<(usize, usize)> =
            plan.chunks.iter().map(|c| (c.slot_vars, c.count)).collect();
        assert_eq!(
            cuts,
            vec![(1, 1), (0, 1), (0, 1), (1, 3), (0, 7), (1, 5), (2, 3)],
            "cap {cap}: every fan cut"
        );
        let proof = prove(&committed, &cfg, ProverFaults::default());
        assert_eq!(proof.argue.bus_outputs.len(), plan.trees());
        assert_eq!(plan.trees(), 3 + 3 + 7 + 5 + 3);
        verify(&committed, &proof, &cfg).unwrap_or_else(|e| panic!("cap {cap}: {e:?}"));
    }
}

/// The fixture the packed tamper tests share: one bin, the fans cut at τ = 2.
fn packed_fixture<'a>(
    airs: &'a Airs,
    fans: &'a [Air<EmptyConstraints>],
    fan_cols: &[Vec<Vec<FE>>],
) -> (CommittedTables<'a, Fp, Ext, KeccakWhir>, Proof, ChainConfig) {
    let cfg = packed(27, 2);
    let committed = commit(packed_tables(airs, fans, fan_cols), &cfg);
    let proof = prove(&committed, &cfg, ProverFaults::default());
    verify(&committed, &proof, &cfg).expect("the untampered packed proof verifies");
    (committed, proof, cfg)
}

fn packed_tampered(tamper: impl FnOnce(&mut Proof)) -> Result<(), MlError> {
    let airs = airs();
    let fans = fan_airs();
    let fan_cols = fan_cols();
    let (committed, mut proof, cfg) = packed_fixture(&airs, &fans, &fan_cols);
    tamper(&mut proof);
    verify(&committed, &proof, &cfg)
}

fn packed_faulted(faults: ProverFaults) -> Result<(), MlError> {
    let airs = airs();
    let fans = fan_airs();
    let fan_cols = fan_cols();
    let cfg = packed(27, 2);
    let committed = commit(packed_tables(&airs, &fans, &fan_cols), &cfg);
    let proof = prove(&committed, &cfg, faults);
    verify(&committed, &proof, &cfg)
}

// ─── neither verifier takes the other's proof ────────────────────────────

/// ★ A packed proof is refused by the unpacked verifier, and the unpacked proof by the
/// packed verifier: the format is the verifier's constant, never the proof's.
#[test]
fn neither_format_verifies_the_others_proof() {
    let airs = airs();
    let fans = fan_airs();
    let fan_cols = fan_cols();
    let today = unpacked(27);
    let pack = packed(27, 2);
    let committed = commit(packed_tables(&airs, &fans, &fan_cols), &today);
    let todays = prove(&committed, &today, ProverFaults::default());
    let packs = prove(&committed, &pack, ProverFaults::default());
    verify(&committed, &todays, &today).expect("the unpacked proof, the unpacked verifier");
    verify(&committed, &packs, &pack).expect("the packed proof, the packed verifier");
    assert_eq!(
        verify(&committed, &packs, &today),
        Err(MlError::ArgueShapeMismatch { part: "tables" }),
        "the unpacked verifier refuses a packed proof"
    );
    assert_eq!(
        verify(&committed, &todays, &pack),
        Err(MlError::ArgueShapeMismatch { part: "tables" }),
        "the packed verifier refuses the unpacked proof"
    );
    // Another price that cuts the same tables another way is another format.
    let other = packed(27, 0);
    assert_ne!(plan_of(&committed, &other), plan_of(&committed, &pack));
    assert!(
        verify(&committed, &packs, &other).is_err(),
        "τ = 0 refuses τ = 2's proof"
    );
}

/// Where no table is cut, the packed proof is the unpacked proof, byte for byte:
/// one protocol instance, so either verifier takes it.
#[test]
fn with_nothing_cut_the_packed_proof_is_todays() {
    let airs = airs();
    let fans = fan_airs();
    let fan_cols = fan_cols();
    let today = unpacked(27);
    let uncut = packed(27, 40);
    let committed = commit(packed_tables(&airs, &fans, &fan_cols), &today);
    assert!(
        plan_of(&committed, &uncut)
            .chunks
            .iter()
            .all(|c| c.count == 1)
    );
    let todays = prove(&committed, &today, ProverFaults::default());
    let uncuts = prove(&committed, &uncut, ProverFaults::default());
    assert!(
        bincode::serialize(&todays).unwrap() == bincode::serialize(&uncuts).unwrap(),
        "the same bytes"
    );
    verify(&committed, &uncuts, &today).expect("the unpacked verifier");
}

// ─── tampers under the packing ───────────────────────────────────────────

/// A wrong count of trees: a bus output short or extra, or a proof laid out
/// under another price.
#[test]
fn a_wrong_tree_count_is_refused() {
    assert_eq!(
        packed_tampered(|p| {
            p.argue.bus_outputs.pop();
        }),
        Err(MlError::ArgueShapeMismatch { part: "tables" })
    );
    assert_eq!(
        packed_tampered(|p| {
            let first = p.argue.bus_outputs[0];
            p.argue.bus_outputs.push(first);
        }),
        Err(MlError::ArgueShapeMismatch { part: "tables" })
    );
    for tree_log_cells in [0u8, 40] {
        assert!(
            packed_faulted(ProverFaults {
                pack: Some(GkrPack::Chunks { tree_log_cells }),
                ..ProverFaults::default()
            })
            .is_err(),
            "the prover cut at τ = {tree_log_cells}"
        );
    }
    assert!(
        packed_faulted(ProverFaults {
            pack: Some(GkrPack::Off),
            ..ProverFaults::default()
        })
        .is_err(),
        "the prover did not cut"
    );
}

/// A wrong claimed sum: any chunk's output fraction off by one, either side.
#[test]
fn a_wrong_chunk_output_is_refused() {
    let trees = 3 + 3 + 7 + 5 + 3;
    for i in 0..trees {
        assert!(
            packed_tampered(|p| p.argue.bus_outputs[i].0 += ExtE::one()).is_err(),
            "tree {i}'s numerator"
        );
        assert!(
            packed_tampered(|p| p.argue.bus_outputs[i].1 += ExtE::one()).is_err(),
            "tree {i}'s denominator"
        );
    }
}

/// Two chunks' outputs swapped in the proof, or the prover's trees built over
/// its chunks in the wrong order (honest trees, a balanced bus, an honest
/// ladder): the claim then is another layer's, and the constraint phase
/// refuses it.
#[test]
fn swapped_chunks_are_refused() {
    assert!(
        packed_tampered(|p| p.argue.bus_outputs.swap(3, 4)).is_err(),
        "two chunks' outputs swapped"
    );
    for t in 3..7 {
        assert_eq!(
            packed_faulted(ProverFaults {
                swap_chunks: Some(t),
                ..ProverFaults::default()
            }),
            Err(MlError::BatchMismatch),
            "table {t}'s first two chunks swapped"
        );
    }
}

/// A chunk's halves wrong at a mid step: that step's residual names it.
#[test]
fn one_chunks_halves_wrong_are_refused() {
    let airs = airs();
    let fans = fan_airs();
    let fan_cols = fan_cols();
    let (_, proof, _) = packed_fixture(&airs, &fans, &fan_cols);
    let step = 1;
    let active = proof.argue.gkr[0].layers[step].halves.len();
    assert!(active > 7, "the bin's step {step} steps {active} trees");
    for tree in 0..active {
        assert_eq!(
            packed_tampered(|p| p.argue.gkr[0].layers[step].halves[tree].q_lo += ExtE::one()),
            Err(MlError::LayerRelationMismatch { layer: step }),
            "tree {tree}"
        );
    }
}

/// A forged balance on a cut table's first chunk (its numerator the value
/// that makes the bus sum to zero, the ladder run on its true tree): refused
/// by step 0's residual, and accepted with that check off — the residual is
/// as load-bearing under the packing as without it.
#[test]
fn a_balancing_chunk_output_is_refused_by_the_residual_alone() {
    let airs = airs();
    let fans = fan_airs();
    let mut fan_cols = fan_cols();
    let cfg = packed(27, 2);
    // FAN(7)'s odd sender reads the zero column: one row of it set to one
    // unbalances the bus.
    fan_cols[1][1][2] = FE::one();
    let committed = commit(packed_tables(&airs, &fans, &fan_cols), &cfg);
    let honest = prove(&committed, &cfg, ProverFaults::default());
    assert_eq!(
        verify(&committed, &honest, &cfg),
        Err(MlError::BusImbalance),
        "premise: the trace's bus does not balance"
    );
    let forged = prove(
        &committed,
        &cfg,
        ProverFaults {
            balance_with: Some(4),
            ..ProverFaults::default()
        },
    );
    assert_eq!(
        verify(&committed, &forged, &cfg),
        Err(MlError::LayerRelationMismatch { layer: 0 })
    );
    let checks = VerifierChecks {
        residual: false,
        ..VerifierChecks::ALL
    };
    assert_eq!(
        verify_with_checks(&committed, &forged, &cfg, checks),
        Ok(())
    );
}

/// A cut table's claim absorbed off by one is refused, as an uncut table's.
#[test]
fn a_tampered_combined_claim_is_refused() {
    for t in [0, 3, 6] {
        assert!(
            packed_faulted(ProverFaults {
                tamper_claim: Some(t),
                ..ProverFaults::default()
            })
            .is_err(),
            "table {t}"
        );
    }
}

// ─── the card ────────────────────────────────────────────────────────────

/// ★ The card proves the host reference's bytes under the packing, with the
/// transcript in the same state after: tables tall enough for the card's
/// trees (CPU 2^13 rows, the fans 2^13 and 2^15), every fan cut at τ = 12 —
/// a last chunk padded to its slots among them — in one bin and in bins of
/// 2^14 cells. Without a device it is the host against the device path's
/// host fallbacks.
///
/// ```text
/// cargo test --release -p stark --features cuda,multilinear/cuda --lib -- \
///     multilinear_table::batched::tests::pack::the_card_proves_the_host_references_packed_bytes --exact --nocapture
/// ```
#[test]
fn the_card_proves_the_host_references_packed_bytes() {
    use crypto::fiat_shamir::is_transcript::IsTranscript;

    const TALL: [(usize, usize); 4] = [(5, 13), (7, 15), (9, 13), (11, 13)];
    let _switches = card_switches();
    a_present_card_is_reachable();
    let airs = airs();
    let fans: Vec<Air<EmptyConstraints>> = TALL.iter().map(|&(m, _)| fan_air(m)).collect();
    let fan_cols: Vec<Vec<Vec<FE>>> = TALL
        .iter()
        .enumerate()
        .map(|(i, &(_, n))| fan_columns(n, i as u64 + 1))
        .collect();
    let cols = columns(12, 1);
    for cap in [27u8, 14] {
        let cfg = packed(cap, 12);
        let run = |at: Where| {
            let mut all = tables(&airs, &cols, false);
            for (air, cols) in fans.iter().zip(&fan_cols) {
                all.push(table(air, cols));
            }
            let committed = commit(all, &cfg);
            let plan = plan_of(&committed, &cfg);
            assert_eq!(
                plan.chunks[3..]
                    .iter()
                    .map(|c| (c.slot_vars, c.count))
                    .collect::<Vec<_>>(),
                vec![(1, 3), (0, 7), (1, 5), (2, 3)],
                "every fan cut"
            );
            let trees = multilinear::gpu::tree_calls();
            let mut transcript = DefaultTranscript::<Ext>::new(SEED);
            let proof = prove_batched_on(
                &committed,
                &cfg,
                &mut transcript,
                None,
                ProverFaults::default(),
                at,
            )
            .unwrap_or_else(|e| panic!("{at:?}: {e:?}"));
            verify(&committed, &proof, &cfg).unwrap_or_else(|e| panic!("{at:?}: {e:?}"));
            (
                bincode::serialize(&proof).expect("bytes"),
                transcript.state(),
                transcript.sample_field_element(),
                multilinear::gpu::tree_calls() - trees,
            )
        };
        let host = run(Where::Host);
        let card = run(Where::Device);
        eprintln!(
            "packed argue cap={cap}: {} bytes, card {}, {} trees built on it",
            host.0.len(),
            if a_device() {
                "present"
            } else {
                "absent (host fallbacks)"
            },
            card.3
        );
        // (The counter is the process's: a test beside this one that holds no
        // switch lock can move it, so only the card's side is asserted.)
        if a_device() {
            assert!(card.3 > 0, "cap={cap}: the card built trees");
        }
        assert!(host.0 == card.0, "cap={cap}: the canonical bytes differ");
        assert_eq!(host.1, card.1, "cap={cap}: the transcripts parted");
        assert_eq!(host.2, card.2, "cap={cap}: the next challenge moved");
    }
}

/// ★ A bin that hands its trees' input layers back
/// (`LAMBDA_VM_ARGUE_BIN_HAND_BACK`, forced here on every bin) proves the
/// bytes of the bin that carries them, with the transcript in the same state
/// after: the fixture of the test above, every fan cut at τ = 12, in one bin
/// and in bins of 2^14 cells. On a device the trees that handed back are
/// counted, so the comparison is not of the carry with itself.
///
/// ```text
/// cargo test --release -p stark --features cuda,multilinear/cuda --lib -- \
///     multilinear_table::batched::tests::pack::a_bin_that_hands_back_proves_the_carried_bytes --exact --nocapture
/// ```
#[test]
fn a_bin_that_hands_back_proves_the_carried_bytes() {
    use crypto::fiat_shamir::is_transcript::IsTranscript;

    struct Restore;
    impl Drop for Restore {
        fn drop(&mut self) {
            multilinear::gpu::force_bin_hand_back_for_tests(false);
        }
    }
    const TALL: [(usize, usize); 4] = [(5, 13), (7, 15), (9, 13), (11, 13)];
    let _switches = card_switches();
    let _restore = Restore;
    a_present_card_is_reachable();
    let airs = airs();
    let fans: Vec<Air<EmptyConstraints>> = TALL.iter().map(|&(m, _)| fan_air(m)).collect();
    let fan_cols: Vec<Vec<Vec<FE>>> = TALL
        .iter()
        .enumerate()
        .map(|(i, &(_, n))| fan_columns(n, i as u64 + 1))
        .collect();
    let cols = columns(12, 1);
    for cap in [27u8, 14] {
        let cfg = packed(cap, 12);
        let run = |hand_back: bool| {
            multilinear::gpu::force_bin_hand_back_for_tests(hand_back);
            let mut all = tables(&airs, &cols, false);
            for (air, cols) in fans.iter().zip(&fan_cols) {
                all.push(table(air, cols));
            }
            let committed = commit(all, &cfg);
            let handed = multilinear::gpu::handed_back_by_preference();
            let mut transcript = DefaultTranscript::<Ext>::new(SEED);
            let proof = prove_batched_on(
                &committed,
                &cfg,
                &mut transcript,
                None,
                ProverFaults::default(),
                Where::Device,
            )
            .unwrap_or_else(|e| panic!("hand back {hand_back}: {e:?}"));
            multilinear::gpu::force_bin_hand_back_for_tests(false);
            verify(&committed, &proof, &cfg)
                .unwrap_or_else(|e| panic!("hand back {hand_back}: {e:?}"));
            (
                bincode::serialize(&proof).expect("bytes"),
                transcript.state(),
                transcript.sample_field_element(),
                multilinear::gpu::handed_back_by_preference() - handed,
            )
        };
        let carried = run(false);
        let handed = run(true);
        eprintln!(
            "bin hand back cap={cap}: {} bytes, card {}, {} trees handed back",
            carried.0.len(),
            if a_device() {
                "present"
            } else {
                "absent (host fallbacks)"
            },
            handed.3
        );
        assert_eq!(carried.3, 0, "cap={cap}: nothing hands back unasked");
        if a_device() {
            assert!(handed.3 > 0, "cap={cap}: trees handed back");
        }
        assert!(
            carried.0 == handed.0,
            "cap={cap}: the canonical bytes differ"
        );
        assert_eq!(carried.1, handed.1, "cap={cap}: the transcripts parted");
        assert_eq!(carried.2, handed.2, "cap={cap}: the next challenge moved");
    }
}

/// ★ A card tree whose input layer cannot be written again for its last
/// sumcheck (faulted here on every rewrite: the bins forced to hand back, and
/// the padded chunks kept without their padding) has it written on the host
/// and proves the bytes of the bin that carries it, with the transcript in the
/// same state after. Without the host's write the argue would fail there. On
/// a device the layers written on the host are counted, so the fault is seen
/// to fire.
///
/// ```text
/// cargo test --release -p stark --features cuda,multilinear/cuda --lib -- \
///     multilinear_table::batched::tests::pack::a_failed_input_rewrite_is_written_on_the_host --exact --nocapture
/// ```
#[test]
fn a_failed_input_rewrite_is_written_on_the_host() {
    use crypto::fiat_shamir::is_transcript::IsTranscript;

    struct Restore;
    impl Drop for Restore {
        fn drop(&mut self) {
            multilinear::gpu::force_bin_hand_back_for_tests(false);
            multilinear::gpu::fault_input_rewrites_for_tests(false);
        }
    }
    const TALL: [(usize, usize); 4] = [(5, 13), (7, 15), (9, 13), (11, 13)];
    let _switches = card_switches();
    let _restore = Restore;
    a_present_card_is_reachable();
    let airs = airs();
    let fans: Vec<Air<EmptyConstraints>> = TALL.iter().map(|&(m, _)| fan_air(m)).collect();
    let fan_cols: Vec<Vec<Vec<FE>>> = TALL
        .iter()
        .enumerate()
        .map(|(i, &(_, n))| fan_columns(n, i as u64 + 1))
        .collect();
    let cols = columns(12, 1);
    for cap in [27u8, 14] {
        let cfg = packed(cap, 12);
        let run = |faulted: bool| {
            multilinear::gpu::force_bin_hand_back_for_tests(faulted);
            multilinear::gpu::fault_input_rewrites_for_tests(faulted);
            let mut all = tables(&airs, &cols, false);
            for (air, cols) in fans.iter().zip(&fan_cols) {
                all.push(table(air, cols));
            }
            let committed = commit(all, &cfg);
            let written = multilinear::gpu::inputs_written_on_host();
            let mut transcript = DefaultTranscript::<Ext>::new(SEED);
            let proof = prove_batched_on(
                &committed,
                &cfg,
                &mut transcript,
                None,
                ProverFaults::default(),
                Where::Device,
            )
            .unwrap_or_else(|e| panic!("faulted {faulted}: {e:?}"));
            multilinear::gpu::force_bin_hand_back_for_tests(false);
            multilinear::gpu::fault_input_rewrites_for_tests(false);
            verify(&committed, &proof, &cfg).unwrap_or_else(|e| panic!("faulted {faulted}: {e:?}"));
            (
                bincode::serialize(&proof).expect("bytes"),
                transcript.state(),
                transcript.sample_field_element(),
                multilinear::gpu::inputs_written_on_host() - written,
            )
        };
        let carried = run(false);
        let faulted = run(true);
        eprintln!(
            "input rewrite on the host cap={cap}: {} bytes, card {}, {} input layers written on the host",
            carried.0.len(),
            if a_device() {
                "present"
            } else {
                "absent (host fallbacks)"
            },
            faulted.3
        );
        assert_eq!(
            carried.3, 0,
            "cap={cap}: no layer is written on the host unfaulted"
        );
        if a_device() {
            assert!(
                faulted.3 > 0,
                "cap={cap}: the faulted rewrites went to the host"
            );
        }
        assert!(
            carried.0 == faulted.0,
            "cap={cap}: the canonical bytes differ"
        );
        assert_eq!(carried.1, faulted.1, "cap={cap}: the transcripts parted");
        assert_eq!(carried.2, faulted.2, "cap={cap}: the next challenge moved");
    }
}

// ─── mass in a padding slot ──────────────────────────────────────────────

/// ★ Padding is the verifier's to define: an unbalanced trace (FAN(7)'s odd
/// sender reading a one) whose prover moves the bus's whole sum into a padding
/// slot of FAN(5)'s short last chunk — honest trees, outputs and ladder over
/// that list, so the bus balances — is refused by the constraint phase, where
/// the verifier's own `0/1` padding meets the claim; with that check off, it
/// is accepted, so nothing else objects.
#[test]
fn mass_in_a_padding_slot_is_refused_by_the_verifiers_own_padding() {
    let airs = airs();
    let fans = fan_airs();
    let mut fan_cols = fan_cols();
    let cfg = packed(27, 2);
    fan_cols[1][1][2] = FE::one();
    let committed = commit(packed_tables(&airs, &fans, &fan_cols), &cfg);
    let plan = plan_of(&committed, &cfg);
    assert_eq!(
        (plan.chunks[3].slot_vars, plan.chunks[3].count),
        (1, 3),
        "FAN(5): three chunks of two slots, the last holding one interaction"
    );
    let honest = prove(&committed, &cfg, ProverFaults::default());
    assert_eq!(
        verify(&committed, &honest, &cfg),
        Err(MlError::BusImbalance),
        "premise: the trace's bus does not balance"
    );
    let forged = prove(
        &committed,
        &cfg,
        ProverFaults {
            pad_mass: Some(3),
            ..ProverFaults::default()
        },
    );
    assert_eq!(
        verify(&committed, &forged, &cfg),
        Err(MlError::BatchMismatch),
        "the padding slot's mass is refused where the verifier's padding enters"
    );
    let checks = VerifierChecks {
        constraint: false,
        ..VerifierChecks::ALL
    };
    assert_eq!(
        verify_with_checks(&committed, &forged, &cfg, checks),
        Ok(()),
        "with the constraint check off the forgery passes: the bus balances and the ladder holds"
    );
}

// ─── where the combination challenge is drawn ─────────────────────────────

/// What the prover did to its transcript, in order.
#[derive(Clone, Debug, PartialEq)]
enum Event {
    Absorb(ExtE),
    Bytes,
    Draw,
}

/// A transcript that logs every absorb and draw of the one it wraps.
struct Recording {
    inner: DefaultTranscript<Ext>,
    log: Vec<Event>,
}

impl crypto::fiat_shamir::is_transcript::IsTranscript<Ext> for Recording {
    fn append_field_element(&mut self, element: &ExtE) {
        self.log.push(Event::Absorb(*element));
        self.inner.append_field_element(element);
    }
    fn append_bytes(&mut self, new_bytes: &[u8]) {
        self.log.push(Event::Bytes);
        self.inner.append_bytes(new_bytes);
    }
    fn mark_statement_end(&mut self) {
        self.inner.mark_statement_end();
    }
    fn state(&self) -> [u8; 32] {
        self.inner.state()
    }
    fn sample_field_element(&mut self) -> ExtE {
        self.log.push(Event::Draw);
        self.inner.sample_field_element()
    }
    fn sample_u64(&mut self, upper_bound: u64) -> u64 {
        self.log.push(Event::Draw);
        self.inner.sample_u64(upper_bound)
    }
}

impl crypto::fiat_shamir::transcript_hash::HasTranscriptHash for Recording {
    type Hash =
        <DefaultTranscript<Ext> as crypto::fiat_shamir::transcript_hash::HasTranscriptHash>::Hash;
}

/// Where `values` were absorbed one after another in `log`: the index just
/// past the last of them (the run must occur exactly once).
fn just_after(log: &[Event], values: &[ExtE]) -> usize {
    let absorbs: Vec<Event> = values.iter().copied().map(Event::Absorb).collect();
    let at: Vec<usize> = (0..=log.len().saturating_sub(absorbs.len()))
        .filter(|&i| log[i..i + absorbs.len()] == absorbs[..])
        .collect();
    assert_eq!(at.len(), 1, "the run is absorbed once");
    at[0] + absorbs.len()
}

/// The draws in a row from `at` on.
fn draws_from(log: &[Event], at: usize) -> usize {
    log[at..].iter().take_while(|e| **e == Event::Draw).count()
}

/// ★ `s'` is drawn after the bin's whole ladder, never before: drawn before,
/// the prover would know it while fixing its chunk claims, and a false claim
/// would no longer be caught with probability `≥ 1 − ℓ/|E|`. Every round trip
/// and every tamper would still pass with the draw moved earlier on both
/// sides, so this pins its place in the prover's transcript on the small cut
/// fixture (one bin, four cut tables): right after the bus outputs come one
/// draw (the first step's batching challenge) and then the ladder's absorbs;
/// right after the last step's halves come exactly `1 + Σ ℓ + n` draws (the
/// last line challenge, every cut table's `s'`, then the constraint point `ξ`)
/// and then the tables' claims. The verifier replays the same transcript, so
/// the honest proof verifying pins its side too.
#[test]
fn the_combination_challenge_is_drawn_after_the_whole_ladder() {
    let airs = airs();
    let fans = fan_airs();
    let fan_cols = fan_cols();
    let cfg = packed(27, 2);
    let committed = commit(packed_tables(&airs, &fans, &fan_cols), &cfg);
    let plan = plan_of(&committed, &cfg);
    assert_eq!(plan.bins.len(), 1, "one bin");
    let combine: usize = plan
        .chunks
        .iter()
        .filter(|c| c.count > 1)
        .map(Chunking::combine_vars)
        .sum();
    assert_eq!(
        combine,
        2 + 3 + 3 + 2,
        "FAN(5), FAN(7), FAN(9), FAN(11): ℓ = 2, 3, 3, 2"
    );

    let mut transcript = Recording {
        inner: DefaultTranscript::<Ext>::new(SEED),
        log: Vec::new(),
    };
    let proof = prove_batched_with(
        &committed,
        &cfg,
        &mut transcript,
        None,
        ProverFaults::default(),
    )
    .expect("the prover runs");
    verify(&committed, &proof, &cfg).expect("the honest proof verifies");
    let log = transcript.log;

    let outputs: Vec<ExtE> = proof
        .argue
        .bus_outputs
        .iter()
        .flat_map(|(p, q)| [*p, *q])
        .collect();
    let ladder_start = just_after(&log, &outputs);
    assert_eq!(
        draws_from(&log, ladder_start),
        1,
        "after the outputs, the first step's batching challenge alone"
    );
    let last = proof.argue.gkr[0].layers.last().expect("a ladder step");
    let halves: Vec<ExtE> = last
        .halves
        .iter()
        .flat_map(|h| [h.p_lo, h.p_hi, h.q_lo, h.q_hi])
        .collect();
    let ladder_end = just_after(&log, &halves);
    assert_eq!(
        draws_from(&log, ladder_end),
        1 + combine + plan.num_vars,
        "after the last halves: the line challenge, every s', then ξ"
    );
    assert!(
        matches!(
            log[ladder_end + 1 + combine + plan.num_vars],
            Event::Absorb(_)
        ),
        "then the tables' claims are absorbed"
    );
}
