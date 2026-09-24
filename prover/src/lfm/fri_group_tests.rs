//! S3 in the in-guest FRI verifier: the shape from the shared schedule (G1)
//! and the group-layer emitter (G2), design/FRI.md §6, §11.
//!
//! Checked against the host's own artefacts, never against a second model:
//! - the in-guest shape (schedule, layer depths, caps) against the host's
//!   `StarkCaps::for_options` / `FriFormat::schedule` over a sweep of shapes;
//! - the emitted verifier against I-FRI-H's checked-in RPX vectors
//!   (`crypto/stark/tests/vectors/zf_fri/d_proof_rpx_*`: pair, dp, the uneven
//!   `[3, 1, 3]` override, and the two capped Q = 20 formats of REVIEW-FRI F9),
//!   executed, with its permutation count equal to the closed form;
//! - tampers of every value a group opening carries, and the slot check shown
//!   load-bearing (a moved `p₀` executes when, and only when, it is skipped);
//! - the {cap off, auto} × {pair, dp, uneven dp} round-trip matrix on a real
//!   laptop-scale proof (F9), both legs as one program;
//! - RULINGS 13: the rows the emitter emits per group layer, against the DP's
//!   cost-model terms (`stark::fri::schedule`), with every unmodelled row named.

use crypto::merkle_tree::cap::CapPolicy;
use serde_json::Value;
use stark::examples::read_only_memory_logup::LogReadOnlyPublicInputs;
use stark::fri::schedule::{
    FRI_COST_WEIGHTS, FRI_FOLD_XALU_ROWS, FRI_SLOT_SELECT_ROWS, FRI_TWIDDLE_BALU_ROWS,
    fri_leaf_blocks, fri_schedule_by,
};
use stark::merkle_caps::StarkCaps;
use stark::proof::options::{FriMode, FriScheduleOverride, ProofFormat, ProofOptions};
use stark::proof::stark::StarkProof;
use stark::proof::view::StarkProofView;

use crate::tables::types::{FE, FEE, GoldilocksExtension, GoldilocksField};

use super::builder::LfmBuilder;
use super::compiler::{LfmProgram, compile};
use super::executor::execute;
use super::fri::{FriShape, LayerCommitment, LayerOpening, emit_group_layer};
use super::fri_tests::{folding_fixture_with, fri_only_program, host_fri_from, permutations};
use super::instr::Instr;
use super::word::{LfmWord, base_word, ext_word, word_as_ext};

type Gl = GoldilocksField;
type Ext3 = GoldilocksExtension;
type VectorProof = StarkProof<Gl, Ext3, LogReadOnlyPublicInputs<Gl>>;

// =============================================================================
// G1 — the in-guest shape IS the host's layout
// =============================================================================

/// ★ One schedule, one depth, one cap per layer, on both sides: the in-guest
/// `FriShape` (the emitter's program shape) against the host's `StarkCaps` —
/// the function the prover embeds caps with and the verifier checks them with,
/// itself built on the host's `FriFoldLayout` — over every LDE size of
/// interest, both terminals in production (T = 9 base legs, T = 10 LFM
/// proofs), both FRI modes and both cap policies.
#[test]
fn the_in_guest_fri_shape_is_the_hosts_layout() {
    let mut checked = 0usize;
    for (blowup, k) in [(4u8, 7u8), (4, 8), (2, 7)] {
        for queries in [3usize, 24, 110] {
            for cap in [CapPolicy::Off, CapPolicy::Auto] {
                for fri in [FriMode::Pair, FriMode::Dp] {
                    let blowup_log = (blowup as u32).trailing_zeros();
                    for lde_log in (blowup_log + 1)..=25 {
                        let opts = ProofOptions {
                            blowup_factor: blowup,
                            fri_number_of_queries: queries,
                            coset_offset: 3,
                            grinding_factor: 0,
                            fri_final_poly_log_degree: k,
                            format: ProofFormat {
                                merkle_cap: cap,
                                fri_mode: fri,
                                ..ProofFormat::DEFAULT
                            },
                        };
                        let shape = FriShape::from_options(&opts, lde_log);
                        shape.check();
                        let host = StarkCaps::for_options(&opts, lde_log as usize, false)
                            .expect("a row-pair format lays out");
                        let depths: Vec<usize> = (0..shape.num_committed())
                            .map(|j| shape.layer_depth(j))
                            .collect();
                        let caps: Vec<usize> = (0..shape.num_committed())
                            .map(|j| shape.layer_cap(j))
                            .collect();
                        assert_eq!(depths, host.fri_depths, "{opts:?} lde {lde_log}");
                        assert_eq!(caps, host.fri, "{opts:?} lde {lde_log}");
                        assert_eq!(shape.index_bits(), host.trace_depth);
                        assert_eq!(shape.is_legacy(), fri == FriMode::Pair);
                        checked += 1;
                    }
                }
            }
        }
    }
    println!("{checked} shapes: in-guest schedule, depths and caps == the host's");
}

// =============================================================================
// G2 — the emitted verifier on I-FRI-H's RPX vectors
// =============================================================================

fn ext_of(v: &Value) -> FEE {
    let limbs: Vec<u64> = v
        .as_array()
        .expect("an ext value is three limbs")
        .iter()
        .map(|x| x.as_u64().expect("a canonical limb"))
        .collect();
    assert_eq!(limbs.len(), 3);
    FEE::new([FE::from(limbs[0]), FE::from(limbs[1]), FE::from(limbs[2])])
}

/// One checked-in RPX (d) vector: its JSON, its proof, and the in-guest shape
/// the emitter builds for it from the vector's FORMAT (the host generator's
/// own `proof_formats`, never re-spelled here).
struct Vector {
    name: &'static str,
    json: Value,
    proof: VectorProof,
    shape: FriShape,
}

fn rpx_vectors() -> Vec<Vector> {
    stark::fri::vectors::proof_formats()
        .into_iter()
        .map(|(name, format, queries)| {
            let dir = stark::fri::vectors::vectors_dir();
            let stem = format!("d_proof_rpx_{name}");
            let json: Value = serde_json::from_slice(
                &std::fs::read(dir.join(format!("{stem}.json"))).expect("the vector JSON"),
            )
            .expect("valid JSON");
            let bytes = std::fs::read(dir.join(format!("{stem}.rkyv"))).expect("the vector proof");
            let proof: VectorProof =
                rkyv::from_bytes::<VectorProof, rkyv::rancor::Error>(&bytes).expect("rkyv");
            let opts = stark::fri::vectors::proof_options(format, queries);
            let lde_log = json["lde_log"].as_u64().expect("lde_log") as u32;
            let shape = FriShape::from_options(&opts, lde_log);
            Vector {
                name,
                json,
                proof,
                shape,
            }
        })
        .collect()
}

impl Vector {
    /// The arenas [`fri_only_program`] declares: `(ι, p₀(υ), p₀(−υ))` per
    /// query, then the roots, the ζs, the terminal coefficients, the per-query
    /// layer openings and (when capped) the caps.
    fn arenas(&self) -> Vec<Vec<LfmWord>> {
        let queries = self.json["queries_detail"].as_array().expect("queries");
        let mut deep = Vec::new();
        for q in queries {
            deep.push(base_word(FE::from(q["iota"].as_u64().expect("iota"))));
            deep.push(ext_word(&ext_of(&q["deep"])));
            deep.push(ext_word(&ext_of(&q["deep_sym"])));
        }
        let view = StarkProofView::Owned(&self.proof);
        let (openings, caps) = super::epoch_verify_tests::fri_layer_openings(view, self.shape);
        let mut per_query = Vec::new();
        for query in &openings {
            for (values, path) in query {
                per_query.extend(values.iter().map(ext_word));
                per_query.extend(super::proof_arena::commitments_to_arena(path));
            }
        }
        let zetas: Vec<LfmWord> = self.json["zetas"]
            .as_array()
            .expect("zetas")
            .iter()
            .map(|z| ext_word(&ext_of(z)))
            .collect();
        let mut out = vec![
            deep,
            super::proof_arena::commitments_to_arena(&self.proof.fri_layers_merkle_roots),
            zetas,
            self.proof
                .fri_final_poly_coeffs
                .iter()
                .map(ext_word)
                .collect(),
            per_query,
        ];
        if self.shape.cap_words(super::proof_arena::words_per_root()) > 0 {
            out.push(super::proof_arena::commitments_to_arena(&caps));
        }
        out
    }

    fn program(&self) -> LfmProgram {
        fri_only_program(self.shape, self.shape.num_queries)
    }
}

/// ★ The emitted FRI verifier accepts every RPX (d) vector — today's pair
/// proof, the DP schedule, the uneven `[3, 1, 3]` override (the only shape
/// that catches a fold-count off-by-one, REVIEW-FRI F6) and both capped Q = 20
/// formats (F9) — with the vector's schedule, depths and caps derived by the
/// emitter's own shape, and the permutation count exactly the closed form.
#[test]
fn the_emitted_fri_verifier_accepts_every_rpx_vector() {
    for v in rpx_vectors() {
        let s = v.shape;
        s.check();
        let schedule: Vec<u8> = v.json["schedule"]
            .as_array()
            .expect("schedule")
            .iter()
            .map(|d| d.as_u64().expect("d") as u8)
            .collect();
        assert_eq!(s.schedule(), schedule, "{}: the schedule", v.name);
        assert_eq!(
            s.is_legacy(),
            v.json["legacy_encoding"]
                .as_bool()
                .expect("legacy_encoding"),
            "{}",
            v.name
        );
        if let Some(caps) = v.json.get("fri_caps") {
            let want: Vec<usize> = caps
                .as_array()
                .expect("fri_caps")
                .iter()
                .map(|c| c.as_u64().expect("c") as usize)
                .collect();
            let depths: Vec<usize> = v.json["fri_tree_depths"]
                .as_array()
                .expect("depths")
                .iter()
                .map(|c| c.as_u64().expect("d") as usize)
                .collect();
            assert_eq!(
                (0..s.num_committed())
                    .map(|j| s.layer_cap(j))
                    .collect::<Vec<_>>(),
                want,
                "{}: caps",
                v.name
            );
            assert_eq!(
                (0..s.num_committed())
                    .map(|j| s.layer_depth(j))
                    .collect::<Vec<_>>(),
                depths,
                "{}: depths",
                v.name
            );
        }
        let program = v.program();
        let exec = execute(&program, &v.arenas(), &crate::hash_pin::BLOCK_HASHER)
            .unwrap_or_else(|e| panic!("{}: the honest vector must execute: {e:?}", v.name));
        assert_eq!(exec.public_words.len(), s.num_queries);
        let closed = s.num_queries * s.permutations_per_query() + s.cap_permutations();
        assert_eq!(
            permutations(&program),
            closed,
            "{}: emitted permutations against the closed form",
            v.name
        );
        println!(
            "{:<9} Q={:<2} schedule {:?} caps {:?}: {} permutations, {} instructions",
            v.name,
            s.num_queries,
            s.schedule(),
            (0..s.num_committed())
                .map(|j| s.layer_cap(j))
                .collect::<Vec<_>>(),
            closed,
            program.instrs.len()
        );
    }
}

/// ★ Every value a group opening carries is bound: the slot value, a non-slot
/// value, the last value of the group, a sibling, a cap word, a folding
/// challenge, a terminal coefficient, and the DEEP value the first slot check
/// compares against. Run on the uneven override and on the capped DP vector.
#[test]
fn no_tampered_group_opening_value_can_pass() {
    for v in rpx_vectors() {
        if !matches!(v.name, "dp_3_1_3" | "cap_dp") {
            continue;
        }
        let program = v.program();
        let honest = v.arenas();
        execute(&program, &honest, &crate::hash_pin::BLOCK_HASHER).expect("honest");
        let q0 = &v.json["queries_detail"][0]["layers"][0];
        let slot = q0["slot"].as_u64().expect("slot") as usize;
        let d0 = 1usize << v.shape.layer_fold(0);
        let other = (slot + 1) % d0;
        // Arenas: deep, roots, zetas, coeffs, queries[, caps].
        let mut bump: Vec<(String, usize, usize)> = vec![
            ("p0 (the DEEP value)".into(), 0, 1),
            ("zeta_1".into(), 2, 1),
            ("terminal coefficient 0".into(), 3, 0),
            (format!("query 0 layer 0 slot value (slot {slot})"), 4, slot),
            (format!("query 0 layer 0 non-slot value {other}"), 4, other),
            ("query 0 layer 0 last group value".into(), 4, d0 - 1),
            ("query 0 layer 0 first sibling".into(), 4, d0),
        ];
        if honest.len() == 6 {
            bump.push(("cap word 0".into(), 5, 0));
            bump.push(("last cap word".into(), 5, honest[5].len() - 1));
        }
        for (label, arena, word) in bump {
            let mut bad = honest.clone();
            bad[arena][word][0] += FE::one();
            execute(&program, &bad, &crate::hash_pin::BLOCK_HASHER).expect_err(&format!(
                "{}: moving {label} must make the program unexecutable",
                v.name
            ));
        }
    }
}

/// ★ The slot check is LOAD-BEARING (the in-guest M1). Under the group
/// encoding the value the first fold produces from the DEEP pair meets the
/// committed layers ONLY at the slot check: the leaf hashes the group, the
/// walk authenticates it, the group fold reads it. So a moved `p₀(υ)` is
/// refused with the check and ACCEPTED without it — which is exactly a
/// verifier that would accept FRI for a different codeword than the trace
/// openings commit to.
#[test]
fn the_slot_check_is_load_bearing() {
    let v = rpx_vectors()
        .into_iter()
        .find(|v| v.name == "cap_dp")
        .expect("the capped dp vector");
    let honest = v.arenas();
    let mut moved = honest.clone();
    moved[0][1][0] += FE::one();

    let with = v.program();
    execute(&with, &honest, &crate::hash_pin::BLOCK_HASHER).expect("honest");
    execute(&with, &moved, &crate::hash_pin::BLOCK_HASHER)
        .expect_err("a moved p0 must be refused by the slot check");

    super::fri::SKIP_SLOT_CHECK.with(|c| c.set(true));
    let without = v.program();
    super::fri::SKIP_SLOT_CHECK.with(|c| c.set(false));
    execute(&without, &moved, &crate::hash_pin::BLOCK_HASHER)
        .expect("WITHOUT the slot check a moved p0 is accepted — the check is the only binding");
}

// =============================================================================
// F9 — the {cap} × {fri} round-trip matrix, both legs, on a real proof
// =============================================================================

/// ★ REVIEW-FRI F9's matrix on a real laptop-scale proof (L2G_MEMORY, 2048
/// rows, blowup 2, `k = 2` so the committed chain covers 11 → 3, Q = 24):
/// {cap off, auto} × {pair, dp, dp `[3, 1, 4]`}. Per cell the FRI leg alone
/// and both legs as one program execute over every query, reach the terminal
/// codeword production computed, and emit exactly the closed form.
#[test]
fn the_cap_and_fri_matrix_round_trips_in_guest() {
    use super::epoch_verify::{blocks_for, group_leaf_felts};

    let hash = super::edsl::WrapHash::production();
    for cap in [CapPolicy::Off, CapPolicy::Auto] {
        for (label, fri, over) in [
            ("pair", FriMode::Pair, None),
            ("dp", FriMode::Dp, None),
            ("dp [3,1,4]", FriMode::Dp, Some(&[3u8, 1, 4][..])),
        ] {
            let mut opts = stark::proof::options::GoldilocksCubicProofOptions::with_blowup(2)
                .expect("blowup 2");
            opts.fri_number_of_queries = 24;
            opts.grinding_factor = 0;
            opts.fri_final_poly_log_degree = 2;
            opts.format = ProofFormat {
                merkle_cap: cap,
                fri_mode: fri,
                fri_schedule_override: over.and_then(FriScheduleOverride::new),
                ..ProofFormat::DEFAULT
            };
            let (air, proof) = folding_fixture_with(2048, opts);
            let h = host_fri_from(&*air, &proof);
            let s = h.shape;
            let all: Vec<usize> = (0..h.trace.iotas.len()).collect();
            let codeword = h.terminal_codeword();
            let position = |iota: usize| iota >> (s.total_folds() - 1);

            // The FRI leg alone.
            let program = fri_only_program(s, all.len());
            let exec = execute(
                &program,
                &h.all_arenas(&all),
                &crate::hash_pin::BLOCK_HASHER,
            )
            .unwrap_or_else(|e| panic!("cap={cap} fri={label}: FRI leg: {e:?}"));
            for (k, &q) in all.iter().enumerate() {
                let v = word_as_ext(&exec.public_words[k].1).expect("ext");
                assert_eq!(
                    v,
                    codeword[position(h.trace.iotas[q])],
                    "cap={cap} fri={label}"
                );
            }
            assert_eq!(
                permutations(&program),
                all.len() * s.permutations_per_query() + s.cap_permutations(),
                "cap={cap} fri={label}: FRI leg closed form"
            );

            // Both legs as one program.
            let mut b = LfmBuilder::new().with_wrap_hash(hash);
            let (_, _, terminal) =
                super::fri::emit_sub_proof_with_fri(&mut b, &h.trace.shape, s, all.len());
            for t in &terminal {
                b.public(t.as_cell());
            }
            let joined = compile(b.finish());
            let mut arenas = h.trace.arenas(&all);
            arenas.extend(h.fri_arenas(&all));
            let exec = execute(&joined, &arenas, &crate::hash_pin::BLOCK_HASHER)
                .unwrap_or_else(|e| panic!("cap={cap} fri={label}: joined: {e:?}"));
            for (k, &q) in all.iter().enumerate() {
                let v = word_as_ext(&exec.public_words[k].1).expect("ext");
                assert_eq!(v, codeword[position(h.trace.iotas[q])]);
            }
            let sub = &h.trace.shape;
            let leaves: usize = sub
                .groups()
                .iter()
                .map(|g| blocks_for(group_leaf_felts(g), hash))
                .sum();
            let closed = all.len()
                * (leaves + sub.groups().len() * sub.path_len() + s.permutations_per_query())
                + sub.cap_permutations()
                + s.cap_permutations();
            assert_eq!(
                permutations(&joined),
                closed,
                "cap={cap} fri={label}: both legs' closed form"
            );
            println!(
                "cap={cap:<4} fri={label:<10} schedule {:?} FRI caps {:?} trace cap {}: FRI leg \
                 {} perms, both legs {} perms / {} instructions",
                s.schedule(),
                (0..s.num_committed())
                    .map(|j| s.layer_cap(j))
                    .collect::<Vec<_>>(),
                sub.trace_cap,
                permutations(&program),
                closed,
                joined.instrs.len(),
            );
        }
    }
}

// =============================================================================
// RULINGS 13 — the emitted rows per group layer against the DP's cost terms
// =============================================================================

/// Rows one group layer of fold exponent `d` emits, by kind, measured on the
/// emitter itself: the layer is emitted TWICE in one builder over hinted
/// inputs and the second emission is counted, so interned program constants
/// (paid once per program) are out of the figure. The tree is two levels
/// deep and uncapped, which isolates the model's path term.
struct LayerRows {
    selects: usize,
    xalu: usize,
    balu: usize,
    hashes: usize,
    unpacks: usize,
    hints: usize,
    total: usize,
}

fn measure_group_layer(d: u32) -> LayerRows {
    let once = group_layer_program(d, 1);
    let twice = group_layer_program(d, 2);
    let (a, b) = (count_kinds(&once.instrs), count_kinds(&twice.instrs));
    LayerRows {
        selects: b.0 - a.0,
        xalu: b.1 - a.1,
        balu: b.2 - a.2,
        hashes: b.3 - a.3,
        unpacks: b.4 - a.4,
        hints: b.5 - a.5,
        total: twice.instrs.len() - once.instrs.len(),
    }
}

/// A program emitting `times` group layers of exponent `d` over hinted
/// inputs that are all hinted BEFORE the first emission, so the difference
/// between `times = 2` and `times = 1` is exactly one layer's rows. One
/// committed layer over a two-level tree: `n − 1 = d + 2` index bits and a
/// terminal at `2^2` (blowup `2^1`, `k = 1`).
fn group_layer_program(d: u32, times: usize) -> LfmProgram {
    let shape = FriShape {
        log2_lde_length: d + 3,
        blowup_log: 1,
        final_poly_log_degree: 1,
        coset_offset: 3,
        num_queries: 1,
        format: ProofFormat {
            fri_mode: FriMode::Dp,
            fri_schedule_override: FriScheduleOverride::new(&[d as u8]),
            ..ProofFormat::DEFAULT
        },
    };
    shape.check();
    assert_eq!(shape.schedule(), vec![d as u8]);
    assert_eq!(shape.layer_depth(0), 2);

    let n = 1usize << d;
    let mut b = LfmBuilder::new().with_wrap_hash(super::edsl::WrapHash::production());
    let arena = b.declare_arena((4 + d as usize + times * (n + 2)) as u32);
    let root = b.hint_word(arena, 0);
    let commitment = LayerCommitment::from_lanes(vec![b.unpack(root)]);
    let v = b.hint_word(arena, 1).as_ext();
    let y_inv = b.hint_felt(arena, 2);
    let index = b.hint_felt(arena, 3);
    let bits = b.bit_dec(index, shape.index_bits());
    let zetas: Vec<_> = (0..d).map(|i| b.hint_word(arena, 4 + i).as_ext()).collect();
    let mut at = 4 + d;
    let openings: Vec<LayerOpening> = (0..times)
        .map(|_| {
            let values = (0..n)
                .map(|_| {
                    at += 1;
                    b.hint_word(arena, at - 1).as_ext()
                })
                .collect();
            let siblings = (0..2)
                .map(|_| {
                    at += 1;
                    super::edsl::WrapDigest::from_cell(b.hint_word(arena, at - 1))
                })
                .collect();
            LayerOpening { values, siblings }
        })
        .collect();
    for opening in &openings {
        emit_group_layer(
            &mut b,
            shape,
            0,
            &commitment,
            &zetas,
            v,
            y_inv,
            opening,
            &bits,
        );
    }
    compile(b.finish())
}

/// `(selects, XALU, BALU, hashes, unpacks, hints)` over an instruction list.
fn count_kinds(instrs: &[Instr]) -> (usize, usize, usize, usize, usize, usize) {
    let mut k = (0, 0, 0, 0, 0, 0);
    for i in instrs {
        match i {
            Instr::Select { .. } => k.0 += 1,
            Instr::ExtAlu { .. } => k.1 += 1,
            Instr::BaseAlu { .. } => k.2 += 1,
            Instr::Hash { .. } => k.3 += 1,
            Instr::Unpack { .. } => k.4 += 1,
            Instr::Hint { .. } => k.5 += 1,
            _ => {}
        }
    }
    k
}

/// ★ RULINGS 13: the rows the emitter emits per group layer, against the
/// DP's cost-model terms (I-FRI-H's weights, `stark::fri::schedule`):
///
/// ```text
///   model, per query per committed layer of exponent d over a depth-D tree:
///     leaf(d)·compress + D·(compress + select) + (2^d − 1)·select
///     + (2^d − 1)·fold(5 XALU) + d·twiddle(1 BALU)
/// ```
///
/// The three terms the ruling names — the slot mux, the group fold and the
/// twiddle chain — each MATCH the emitter row for row (and so do the leaf and
/// the walk). The emitter ALSO emits rows the model does not price, and this
/// test pins them rather than hiding them, because the schedule is a format
/// constant and a change of weights is the lead's ruling (RULINGS 13):
///
/// - `x_g⁻¹ = y⁻¹·ω^{br(slot)}`: `d` selects of constants and `d` base muls;
/// - fold-level scaling: one `emul_base` per level with more than two pairs
///   (`max(0, d − 2)` XALU) and one base mul on the level with two pairs
///   (`[d ≥ 2]` BALU);
/// - the slot check's `assert_eq_ext`: 2 XALU;
/// - the per-opening root (or cap node) compare: 8 BALU rows (four lowered
///   asserts) and one unpack — which today's pair layer pays as well;
/// - the group's `2^d` unpacks (the leaf reads three lanes of each value) and
///   `2^d` value hints, plus the walked root's one unpack and the path hints.
///
/// At `d = 1` the model is today's pair layer exactly (1 select, 5 XALU,
/// 1 BALU); the group encoding at `d = 1` pays the extras on top.
#[test]
fn the_group_layer_rows_against_the_dp_cost_model() {
    let w = FRI_COST_WEIGHTS;
    let depth = 2usize;
    println!(
        "\n  d | model sel/XALU/BALU/hash | emitted sel/XALU/BALU/hash | unmodelled \
         sel/XALU/BALU  unpack hint | model ns  unmodelled ns"
    );
    for d in 1..=6u32 {
        let r = measure_group_layer(d);
        let n = 1usize << d;
        // The model's rows (the ruling's terms plus the leaf and the walk).
        let m_sel = (n - 1) * FRI_SLOT_SELECT_ROWS as usize + depth;
        let m_xalu = (n - 1) * FRI_FOLD_XALU_ROWS as usize;
        let m_balu = d as usize * FRI_TWIDDLE_BALU_ROWS as usize;
        let m_hash = fri_leaf_blocks(d) as usize + depth;
        // What the emitter adds on top, by construction (see the doc).
        let x_sel = d as usize;
        let x_xalu = 2 + (d as usize).saturating_sub(2);
        // + the per-opening root compare: four lowered base asserts (a `sub`
        // and a `div` each), today's pair layer pays it too.
        let x_balu = d as usize + usize::from(d >= 2) + 8;
        assert_eq!(
            r.hashes, m_hash,
            "d={d}: leaf blocks + one compression per level"
        );
        assert_eq!(r.selects, m_sel + x_sel, "d={d}: selects");
        assert_eq!(r.xalu, m_xalu + x_xalu, "d={d}: XALU rows");
        assert_eq!(r.balu, m_balu + x_balu, "d={d}: BALU rows");
        assert_eq!(
            r.unpacks,
            n + 1,
            "d={d}: the group's unpacks and the walked root's"
        );
        assert_eq!(r.hints, n + depth, "d={d}: the group's values and its path");
        let model_ns = m_sel as u64 * w.cap.select
            + (n as u64 - 1) * w.fold
            + d as u64 * w.twiddle
            + m_hash as u64 * w.cap.compress;
        let unmodelled_ns = x_sel as u64 * w.cap.select
            + x_xalu as u64 * XALU_NS
            + x_balu as u64 * BALU_NS
            + (n as u64 + 1) * w.cap.unpack
            + n as u64 * w.cap.hint;
        println!(
            "  {d} | {m_sel:>3}/{m_xalu:>4}/{m_balu:>2}/{m_hash:>2}          | \
             {:>3}/{:>4}/{:>2}/{:>2}            | {x_sel:>3}/{x_xalu:>4}/{x_balu:>2}     \
             {:>4} {:>4} | {model_ns:>8} {unmodelled_ns:>8}  ({} instructions)",
            r.selects,
            r.xalu,
            r.balu,
            r.hashes,
            n + 1,
            n,
            r.total,
        );
    }

    // What the unmodelled rows would do to the schedule, for the lead: the DP
    // re-run with them priced (hint words priced at the cap policy's hint
    // weight), at the production terminals and Q = 110 under cap = auto.
    // Printed, not asserted: changing the objective is a format change.
    let cap = CapPolicy::Auto;
    let q = 110u64;
    let with_extras = |d: u32, depth: u32| -> u64 {
        let base = stark::fri::schedule::fri_layer_cost_q(&w, d, depth, q, cap);
        let n = 1u64 << d;
        let extra = u64::from(d) * w.cap.select
            + (2 + u64::from(d.saturating_sub(2))) * XALU_NS
            + (u64::from(d) + u64::from(d >= 2) + 8) * BALU_NS
            + (n + 1) * w.cap.unpack
            + n * w.cap.hint;
        base + q * extra
    };
    println!("\n  schedules at Q = 110, cap = auto: the ruled objective vs the emitted rows");
    for t in [9u32, 10] {
        for b0 in [13u32, 18, 20, 21, 23] {
            let ruled = fri_schedule_by(b0, t, 6, &|d, depth| {
                stark::fri::schedule::fri_layer_cost_q(&w, d, depth, q, cap)
            });
            let emitted = fri_schedule_by(b0, t, 6, &with_extras);
            println!(
                "    T={t} b0={b0}: ruled {:?} (ns·Q {})  |  with the emitted rows {:?} \
                 (ns·Q {})",
                ruled.schedule, ruled.cost_q, emitted.schedule, emitted.cost_q
            );
        }
    }
}

/// Cost-law prices of an `XALU` and a `BALU` row, the schedule module's.
const XALU_NS: u64 = stark::fri::schedule::XALU_ROW_NS;
const BALU_NS: u64 = stark::fri::schedule::BALU_ROW_NS;
