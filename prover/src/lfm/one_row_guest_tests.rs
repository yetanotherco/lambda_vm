//! S2 in the in-guest (LFM) STARK verifier (G3): one-row trace leaves, DEEP
//! at ONE point, the committed FRI input and
//! its input-slot check, index bits over the whole LDE, no `−υ` point.
//!
//! Checked against the host's own artefacts, never against a second model:
//! - the in-guest one-row shape (index bits, schedule, layer depths, caps,
//!   challenge count) against the host's `StarkCaps::for_options(.., true)`;
//! - the emitted FRI verifier against the host's checked-in RPX (e) proofs
//!   (`crypto/stark/tests/vectors/zf_fri/e_proof_rpx_*`), executed, with its
//!   permutation count equal to the closed form;
//! - the in-guest one-row (and row-pair) trace leaf against the (e) leaf
//!   digests;
//! - tampers of every value the input tree's opening carries, and the
//!   input-slot check shown load-bearing (a moved `DEEP(x_r)` executes when,
//!   and only when, the slot check is skipped);
//! - both legs as one program on real laptop-scale proofs at `one_row` ∈
//!   {1, auto} × cap {off, auto} × fri {pair, dp}, and one program verifying a
//!   one-row table and a row-pair table side by side (mixed layouts).

use crypto::merkle_tree::cap::CapPolicy;
use math::fft::bit_reversing::reverse_index;
use serde_json::Value;
use stark::config::Commitment;
use stark::examples::read_only_memory_logup::LogReadOnlyPublicInputs;
use stark::leaf_layout::LeafLayout;
use stark::merkle_caps::StarkCaps;
use stark::proof::options::{FriMode, FriScheduleOverride, OneRowMode, ProofFormat, ProofOptions};
use stark::proof::stark::StarkProof;
use stark::proof::view::StarkProofView;

use crate::tables::types::{FE, FEE, GoldilocksExtension, GoldilocksField};

use super::builder::LfmBuilder;
use super::compiler::{LfmProgram, compile};
use super::epoch_verify::{blocks_for, group_leaf_felts_at};
use super::executor::execute;
use super::fri::FriShape;
use super::fri_tests::{
    HostFri, folding_fixture_with, fri_only_program, host_fri_from, permutations,
};
use super::sub_proof::{GroupShape, emit_leaf_hash_rows};
use super::word::{LfmWord, base_word, ext_word, word_as_ext};

type Gl = GoldilocksField;
type Ext3 = GoldilocksExtension;
type VectorProof = StarkProof<Gl, Ext3, LogReadOnlyPublicInputs<Gl>>;

// =============================================================================
// The shape — the in-guest one-row layout IS the host's
// =============================================================================

/// ★ One index width, one schedule, one depth and one cap per layer on both
/// sides under one-row leaves: the in-guest `FriShape` (resolved to one row)
/// against the host's `StarkCaps` at `one_row = true` — built on the host's
/// `FriFoldLayout` — over every LDE size of interest, both production
/// terminals, both FRI modes and both cap policies. Also: the index is
/// `log2(lde)` bits wide (not `log2(lde) − 1`), the chain covers EVERY fold
/// (layer 0 is the DEEP codeword), and a proof draws one challenge per
/// committed layer (none before the input tree).
#[test]
fn the_in_guest_one_row_shape_is_the_hosts_layout() {
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
                                one_row: OneRowMode::On,
                                ..ProofFormat::DEFAULT
                            },
                        };
                        let shape = FriShape::from_options(&opts, lde_log);
                        shape.check();
                        assert!(shape.one_row());
                        assert!(!shape.is_legacy(), "a one-row table is never legacy");
                        let host = StarkCaps::for_options(&opts, lde_log as usize, true)
                            .expect("a one-row format lays out");
                        assert_eq!(shape.index_bits(), lde_log as usize);
                        assert_eq!(shape.index_bits(), host.trace_depth);
                        let depths: Vec<usize> = (0..shape.num_committed())
                            .map(|j| shape.layer_depth(j))
                            .collect();
                        let caps: Vec<usize> = (0..shape.num_committed())
                            .map(|j| shape.layer_cap(j))
                            .collect();
                        assert_eq!(depths, host.fri_depths, "{opts:?} lde {lde_log}");
                        assert_eq!(caps, host.fri, "{opts:?} lde {lde_log}");
                        let covered: u32 = shape.schedule().iter().map(|&d| u32::from(d)).sum();
                        assert_eq!(covered, shape.total_folds(), "every fold is committed");
                        assert_eq!(
                            shape.num_zetas(),
                            if shape.total_folds() > 0 {
                                shape.num_committed()
                            } else {
                                0
                            }
                        );
                        checked += 1;
                    }
                }
            }
        }
    }
    println!(
        "{checked} one-row shapes: in-guest index bits, schedule, depths and caps == the host's"
    );
}

/// `one_row = auto` has no layout of its own: a shape is built at the table's
/// RESOLVED layout (the AIR's widths decide), and asking the options alone is
/// refused rather than guessed.
#[test]
#[should_panic(expected = "one_row = auto resolves per table")]
fn an_unresolved_auto_layout_is_refused() {
    let mut opts =
        stark::proof::options::GoldilocksCubicProofOptions::with_blowup(4).expect("blowup 4");
    opts.format.one_row = OneRowMode::Auto;
    let _ = FriShape::from_options(&opts, 12);
}

// =============================================================================
// (e) — the emitted FRI verifier on the host's one-row RPX proofs
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

fn commitment_of_hex(s: &str) -> Commitment {
    assert_eq!(s.len(), 64, "a 32-byte digest");
    let mut out = [0u8; 32];
    for (i, byte) in out.iter_mut().enumerate() {
        *byte = u8::from_str_radix(&s[2 * i..2 * i + 2], 16).expect("hex");
    }
    out
}

/// One checked-in RPX (e) proof: its JSON, its proof, and the in-guest shape
/// built from the vector's FORMAT (the host generator's own
/// `one_row_proof_formats`, never re-spelled here).
struct OneRowVector {
    name: &'static str,
    json: Value,
    proof: VectorProof,
    shape: FriShape,
}

fn one_row_rpx_vectors() -> Vec<OneRowVector> {
    stark::fri::vectors::one_row_proof_formats()
        .into_iter()
        .map(|(name, format)| {
            let dir = stark::fri::vectors::vectors_dir();
            let stem = format!("e_proof_rpx_{name}");
            let json: Value = serde_json::from_slice(
                &std::fs::read(dir.join(format!("{stem}.json"))).expect("the vector JSON"),
            )
            .expect("valid JSON");
            let bytes = std::fs::read(dir.join(format!("{stem}.rkyv"))).expect("the vector proof");
            let proof: VectorProof =
                rkyv::from_bytes::<VectorProof, rkyv::rancor::Error>(&bytes).expect("rkyv");
            let queries = json["queries"].as_u64().expect("queries") as usize;
            let opts = stark::fri::vectors::proof_options(format, queries);
            let lde_log = json["lde_log"].as_u64().expect("lde_log") as u32;
            let shape = FriShape::from_options(&opts, lde_log);
            OneRowVector {
                name,
                json,
                proof,
                shape,
            }
        })
        .collect()
}

impl OneRowVector {
    /// The arenas [`fri_only_program`] declares for a one-row shape:
    /// `(r, DEEP(x_r))` per query, then the roots (the input tree's first),
    /// the ζs (one per layer), the terminal coefficients and the per-query
    /// layer openings (every layer a full group — layer 0 the input group).
    fn arenas(&self) -> Vec<Vec<LfmWord>> {
        let queries = self.json["queries_detail"].as_array().expect("queries");
        let mut deep = Vec::new();
        for q in queries {
            deep.push(base_word(FE::from(q["iota"].as_u64().expect("r"))));
            deep.push(ext_word(&ext_of(&q["deep"])));
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

/// ★ The emitted FRI verifier accepts every one-row RPX (e) proof — the pair
/// schedule from the input tree and the uneven `[3, 2, 1, 2]` override — with
/// the shape's index width, schedule, challenge count and layer depths equal
/// to the vector's, and the permutation count exactly the closed form.
#[test]
fn the_emitted_fri_verifier_accepts_every_one_row_rpx_vector() {
    let vectors = one_row_rpx_vectors();
    assert_eq!(vectors.len(), 2, "one_row_pair and one_row_3_2_1_2");
    for v in vectors {
        let s = v.shape;
        s.check();
        assert!(v.json["one_row"].as_bool().expect("one_row"));
        assert!(s.one_row());
        let schedule: Vec<u8> = v.json["schedule"]
            .as_array()
            .expect("schedule")
            .iter()
            .map(|d| d.as_u64().expect("d") as u8)
            .collect();
        assert_eq!(s.schedule(), schedule, "{}: the schedule", v.name);
        assert_eq!(
            s.is_legacy(),
            v.json["legacy_encoding"].as_bool().expect("legacy"),
            "{}",
            v.name
        );
        assert_eq!(
            s.index_bits() as u64,
            v.json["trace_tree_depth"].as_u64().expect("depth"),
            "{}: r has log2(lde) bits",
            v.name
        );
        assert_eq!(
            1u64 << s.index_bits(),
            v.json["query_bound"].as_u64().expect("bound"),
            "{}: r ranges over the whole LDE",
            v.name
        );
        assert_eq!(
            s.num_zetas(),
            v.json["zetas"].as_array().expect("zetas").len(),
            "{}: one challenge per committed layer, none before the input tree",
            v.name
        );
        assert_eq!(s.num_committed(), v.proof.fri_layers_merkle_roots.len());
        for (qi, q) in v.json["queries_detail"]
            .as_array()
            .expect("queries")
            .iter()
            .enumerate()
        {
            for (j, layer) in q["layers"].as_array().expect("layers").iter().enumerate() {
                assert_eq!(
                    s.layer_path_len(j) as u64,
                    layer["path_len"].as_u64().expect("path_len"),
                    "{} query {qi} layer {j}",
                    v.name
                );
            }
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
            "{:<16} Q={} index bits {} schedule {:?} zetas {}: {} permutations, {} instructions",
            v.name,
            s.num_queries,
            s.index_bits(),
            s.schedule(),
            s.num_zetas(),
            closed,
            program.instrs.len()
        );
    }
}

/// ★ Every value the INPUT tree's opening carries is bound, and so is the DEEP
/// value it is checked against: `DEEP(x_r)`, the input root, `ζ₀` (layer 0's
/// challenge under one row), a terminal coefficient, the input group's slot
/// value, a non-slot value, its last value and its first sibling. Run on both
/// (e) formats.
#[test]
fn no_tampered_input_tree_value_can_pass() {
    for v in one_row_rpx_vectors() {
        let program = v.program();
        let honest = v.arenas();
        execute(&program, &honest, &crate::hash_pin::BLOCK_HASHER).expect("honest");
        let q0 = &v.json["queries_detail"][0]["layers"][0];
        let slot = q0["slot"].as_u64().expect("slot") as usize;
        assert_eq!(
            ext_of(&q0["values"][slot]),
            ext_of(&v.json["queries_detail"][0]["deep"]),
            "{}: the input group's slot holds DEEP(x_r) (the host's input-slot check)",
            v.name
        );
        let d0 = 1usize << v.shape.layer_fold(0);
        let other = (slot + 1) % d0;
        // Arenas: deep (r, DEEP) per query, roots, zetas, coeffs, queries.
        let bump: Vec<(String, usize, usize)> = vec![
            ("DEEP(x_r)".into(), 0, 1),
            ("the input root".into(), 1, 0),
            ("zeta_0 (layer 0's challenge)".into(), 2, 0),
            ("terminal coefficient 0".into(), 3, 0),
            (format!("input group slot value (slot {slot})"), 4, slot),
            (format!("input group non-slot value {other}"), 4, other),
            ("input group last value".into(), 4, d0 - 1),
            ("input group first sibling".into(), 4, d0),
        ];
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

/// ★ The INPUT-SLOT check is LOAD-BEARING (the in-guest M1 at the input
/// tree). Under one-row leaves `DEEP(x_r)` meets the committed FRI
/// chain ONLY at `group₀[slot] == DEEP(x_r)`: the input leaf hashes the group,
/// the walk authenticates it, the group fold reads it — none reads `DEEP(x_r)`.
/// So a moved `DEEP(x_r)` is refused with the check and ACCEPTED without it,
/// which is exactly a verifier that would run FRI on a codeword the trace
/// openings do not commit to.
#[test]
fn the_input_slot_check_is_load_bearing() {
    for v in one_row_rpx_vectors() {
        let honest = v.arenas();
        let mut moved = honest.clone();
        moved[0][1][0] += FE::one();

        let with = v.program();
        execute(&with, &honest, &crate::hash_pin::BLOCK_HASHER).expect("honest");
        execute(&with, &moved, &crate::hash_pin::BLOCK_HASHER)
            .expect_err("a moved DEEP(x_r) must be refused by the input-slot check");

        super::fri::SKIP_SLOT_CHECK.with(|c| c.set(true));
        let without = v.program();
        super::fri::SKIP_SLOT_CHECK.with(|c| c.set(false));
        execute(&without, &moved, &crate::hash_pin::BLOCK_HASHER).unwrap_or_else(|e| {
            panic!(
                "{}: WITHOUT the slot check a moved DEEP(x_r) is accepted — the input-slot \
                 check is the only binding: {e:?}",
                v.name
            )
        });
    }
}

// =============================================================================
// (e) — the one-row trace leaf
// =============================================================================

/// ★ The in-guest trace leaf at `rows_per_leaf = 1` (and at 2, today's) is
/// the host's: every leaf of the (e) KAT matrices (16 rows × 5 base
/// columns, 16 rows × 2 ext3 columns, read as bit-reversed LDE columns) under
/// the production hash. One row: leaf `i` = the row at bit-reversed position
/// `i`, columns in order. Row pair: rows `2i` then `2i + 1`.
#[test]
fn the_one_row_trace_leaf_is_the_hosts() {
    let dir = stark::fri::vectors::vectors_dir();
    let json: Value = serde_json::from_slice(
        &std::fs::read(dir.join("e_leaf_digests_rpx.json")).expect("the (e) leaf digests"),
    )
    .expect("valid JSON");
    let rows = json["rows"].as_u64().expect("rows") as usize;
    let base: Vec<Vec<FE>> = json["base_columns"]
        .as_array()
        .expect("base")
        .iter()
        .map(|c| {
            c.as_array()
                .expect("a column")
                .iter()
                .map(|x| FE::from(x.as_u64().expect("a felt")))
                .collect()
        })
        .collect();
    let ext: Vec<Vec<FEE>> = json["ext_columns"]
        .as_array()
        .expect("ext")
        .iter()
        .map(|c| c.as_array().expect("a column").iter().map(ext_of).collect())
        .collect();

    let mut checked = 0usize;
    for layout in json["layouts"].as_array().expect("layouts") {
        let rows_per_leaf = layout["rows_per_leaf"].as_u64().expect("rows") as usize;
        for (is_ext, key, width) in [
            (false, "base_leaves", base.len()),
            (true, "ext_leaves", ext.len()),
        ] {
            let shape = GroupShape {
                num_columns: width,
                is_ext,
            };
            let mut b = LfmBuilder::new().with_wrap_hash(super::edsl::WrapHash::production());
            let arena = b.declare_arena(shape.values_at(rows_per_leaf) as u32);
            let cells: Vec<_> = (0..shape.values_at(rows_per_leaf) as u32)
                .map(|i| b.hint_word(arena, i))
                .collect();
            let leaf = emit_leaf_hash_rows(&mut b, shape, rows_per_leaf, &cells);
            for cell in leaf.cells() {
                b.public(*cell);
            }
            let program = compile(b.finish());

            let leaves = layout[key].as_array().expect("leaves");
            assert_eq!(leaves.len(), rows / rows_per_leaf);
            for (i, want) in leaves.iter().enumerate() {
                let mut words = Vec::new();
                for k in 0..rows_per_leaf {
                    let row = reverse_index(rows_per_leaf * i + k, rows as u64);
                    if is_ext {
                        words.extend(ext.iter().map(|c| ext_word(&c[row])));
                    } else {
                        words.extend(base.iter().map(|c| base_word(c[row])));
                    }
                }
                let exec = execute(&program, &[words], &crate::hash_pin::BLOCK_HASHER)
                    .expect("the leaf hash executes");
                let got: Vec<LfmWord> = exec.public_words.iter().map(|(_, w)| *w).collect();
                assert_eq!(
                    got,
                    super::proof_arena::commitment_words(&commitment_of_hex(
                        want.as_str().expect("hex")
                    )),
                    "rows_per_leaf {rows_per_leaf} {key} leaf {i}"
                );
                checked += 1;
            }
        }
    }
    assert_eq!(checked, 16 * 2 + 8 * 2, "every leaf of both layouts");
    println!("{checked} in-guest trace leaves == the host's (e) digests (rpx)");
}

// =============================================================================
// Round trips on real proofs — both legs as one program
// =============================================================================

fn opts_with(one_row: OneRowMode, cap: CapPolicy, fri: FriMode) -> ProofOptions {
    let mut opts =
        stark::proof::options::GoldilocksCubicProofOptions::with_blowup(2).expect("blowup 2");
    opts.fri_number_of_queries = 24;
    opts.grinding_factor = 0;
    opts.fri_final_poly_log_degree = 2;
    opts.format = ProofFormat {
        merkle_cap: cap,
        fri_mode: fri,
        one_row,
        ..ProofFormat::DEFAULT
    };
    opts
}

/// The terminal-codeword position a query arrives at.
fn terminal_position(s: FriShape, index: usize) -> usize {
    if s.one_row() {
        index >> s.total_folds()
    } else {
        index >> (s.total_folds() - 1)
    }
}

/// Both legs of one real sub-proof emitted into `b`, returning the program's
/// arenas for it (the trace leg's then the FRI leg's) and the closed-form
/// permutation count of the two legs.
fn emit_both_legs(b: &mut LfmBuilder, h: &HostFri) -> (Vec<Vec<LfmWord>>, usize) {
    let s = h.shape;
    let all: Vec<usize> = (0..h.trace.iotas.len()).collect();
    let (_, _, terminal) = super::fri::emit_sub_proof_with_fri(b, &h.trace.shape, s, all.len());
    for t in &terminal {
        b.public(t.as_cell());
    }
    let mut arenas = h.trace.arenas(&all);
    arenas.extend(h.fri_arenas(&all));

    let hash = super::edsl::WrapHash::production();
    let sub = &h.trace.shape;
    let leaves: usize = sub
        .groups()
        .iter()
        .map(|g| blocks_for(group_leaf_felts_at(g, sub.rows_per_leaf()), hash))
        .sum();
    let closed = all.len()
        * (leaves + sub.groups().len() * sub.path_len() + s.permutations_per_query())
        + sub.cap_permutations()
        + s.cap_permutations();
    (arenas, closed)
}

/// ★ The in-guest round trip at `one_row` ∈ {1, auto} × cap {off, auto} × fri
/// {pair, dp} on a real laptop-scale proof (L2G_MEMORY, 2048 rows, blowup 2,
/// `k = 2`, Q = 24): the FRI leg alone and both legs as one program execute
/// over every query, reach the terminal codeword production computed, and emit
/// exactly the closed form. Under `auto` the table's layout is the host's own
/// resolution (printed). At `one_row = 1` a moved one-row trace opening value
/// is refused (the join still binds the fold to the leaf).
#[test]
fn one_row_round_trips_in_guest() {
    let mut layouts = Vec::new();
    for one_row in [OneRowMode::On, OneRowMode::Auto] {
        for cap in [CapPolicy::Off, CapPolicy::Auto] {
            for fri in [FriMode::Pair, FriMode::Dp] {
                let label = format!("one_row={one_row} cap={cap} fri={fri:?}");
                let (air, proof) = folding_fixture_with(2048, opts_with(one_row, cap, fri));
                let h = host_fri_from(&*air, &proof);
                let s = h.shape;
                assert_eq!(h.trace.shape.layout, s.leaf_layout());
                if one_row == OneRowMode::On {
                    assert!(s.one_row(), "{label}");
                }
                layouts.push((label.clone(), s.leaf_layout()));
                let all: Vec<usize> = (0..h.trace.iotas.len()).collect();
                let codeword = h.terminal_codeword();

                // The FRI leg alone.
                let program = fri_only_program(s, all.len());
                let exec = execute(
                    &program,
                    &h.all_arenas(&all),
                    &crate::hash_pin::BLOCK_HASHER,
                )
                .unwrap_or_else(|e| panic!("{label}: FRI leg: {e:?}"));
                for (k, &q) in all.iter().enumerate() {
                    let v = word_as_ext(&exec.public_words[k].1).expect("ext");
                    assert_eq!(
                        v,
                        codeword[terminal_position(s, h.trace.iotas[q])],
                        "{label}"
                    );
                }
                assert_eq!(
                    permutations(&program),
                    all.len() * s.permutations_per_query() + s.cap_permutations(),
                    "{label}: FRI leg closed form"
                );

                // Both legs as one program.
                let mut b = LfmBuilder::new().with_wrap_hash(super::edsl::WrapHash::production());
                let (arenas, closed) = emit_both_legs(&mut b, &h);
                let joined = compile(b.finish());
                let exec = execute(&joined, &arenas, &crate::hash_pin::BLOCK_HASHER)
                    .unwrap_or_else(|e| panic!("{label}: joined: {e:?}"));
                for (k, &q) in all.iter().enumerate() {
                    let v = word_as_ext(&exec.public_words[k].1).expect("ext");
                    assert_eq!(
                        v,
                        codeword[terminal_position(s, h.trace.iotas[q])],
                        "{label}"
                    );
                }
                assert_eq!(
                    permutations(&joined),
                    closed,
                    "{label}: both legs' closed form"
                );

                if s.one_row() {
                    // A one-row trace opening value (query 0, main column 0 —
                    // right after the index word) must not execute.
                    let mut bad = arenas.clone();
                    bad[4][1][0] += FE::one();
                    execute(&joined, &bad, &crate::hash_pin::BLOCK_HASHER).expect_err(&format!(
                        "{label}: a moved one-row trace value must be refused"
                    ));
                    // The upper half of the LDE is reached: r is not a pair index.
                    let lde = 1usize << s.log2_lde_length;
                    assert!(
                        h.trace.iotas.iter().any(|&r| r >= lde / 2),
                        "{label}: 24 one-row indices over the whole LDE reach its upper half"
                    );
                }
                println!(
                    "{label:<34} layout {:?} index bits {} schedule {:?} FRI caps {:?} trace cap \
                     {}: FRI leg {} perms, both legs {} perms / {} instructions",
                    s.leaf_layout(),
                    s.index_bits(),
                    s.schedule(),
                    (0..s.num_committed())
                        .map(|j| s.layer_cap(j))
                        .collect::<Vec<_>>(),
                    h.trace.shape.trace_cap,
                    permutations(&program),
                    closed,
                    joined.instrs.len(),
                );
            }
        }
    }
    assert_eq!(layouts.len(), 8);
}

/// ★ Mixed layouts in ONE program: a one-row table (L2G_MEMORY at 2048 rows,
/// `one_row = 1`, `fri = dp`, cap auto) and a row-pair table (L2G_MEMORY at
/// 1024 rows, today's format) verified side by side, both legs each — the
/// shape of an `auto` epoch whose tables resolve differently. Each table's
/// layout is its own verifier constant; the program executes, every terminal
/// is production's, and the permutations are the sum of the two closed forms.
#[test]
fn a_one_row_and_a_row_pair_table_verify_in_one_program() {
    let (air_a, proof_a) = folding_fixture_with(
        2048,
        opts_with(OneRowMode::On, CapPolicy::Auto, FriMode::Dp),
    );
    let (air_b, proof_b) = folding_fixture_with(
        1024,
        opts_with(OneRowMode::Off, CapPolicy::Off, FriMode::Pair),
    );
    let a = host_fri_from(&*air_a, &proof_a);
    let b_host = host_fri_from(&*air_b, &proof_b);
    assert_eq!(a.shape.leaf_layout(), LeafLayout::Row);
    assert_eq!(b_host.shape.leaf_layout(), LeafLayout::RowPair);

    let mut b = LfmBuilder::new().with_wrap_hash(super::edsl::WrapHash::production());
    let (mut arenas, closed_a) = emit_both_legs(&mut b, &a);
    let (arenas_b, closed_b) = emit_both_legs(&mut b, &b_host);
    arenas.extend(arenas_b);
    let program = compile(b.finish());
    let exec = execute(&program, &arenas, &crate::hash_pin::BLOCK_HASHER)
        .expect("a one-row and a row-pair table verify in one program");
    let mut k = 0usize;
    for h in [&a, &b_host] {
        let codeword = h.terminal_codeword();
        for &iota in &h.trace.iotas {
            let v = word_as_ext(&exec.public_words[k].1).expect("ext");
            assert_eq!(v, codeword[terminal_position(h.shape, iota)]);
            k += 1;
        }
    }
    assert_eq!(permutations(&program), closed_a + closed_b);
    println!(
        "mixed program: one-row table {} perms + row-pair table {} perms = {} ({} instructions)",
        closed_a,
        closed_b,
        closed_a + closed_b,
        program.instrs.len()
    );
}

/// The one-row query index needs a schedule override that the DP never
/// picks to exercise unequal neighbouring exponents from the INPUT tree
/// (the fold-count off-by-one check at layer 0): `[3, 1, 3, 2]` over the 9 committed folds of a
/// 2048-row, blowup-2, `k = 2` one-row table (`12 → 3`, every fold committed)
/// — both legs, executed.
#[test]
fn an_uneven_one_row_schedule_round_trips_in_guest() {
    let mut opts = opts_with(OneRowMode::On, CapPolicy::Off, FriMode::Dp);
    opts.format.fri_schedule_override = FriScheduleOverride::new(&[3, 1, 3, 2]);
    let (air, proof) = folding_fixture_with(2048, opts);
    let h = host_fri_from(&*air, &proof);
    assert_eq!(h.shape.schedule(), vec![3, 1, 3, 2]);
    let mut b = LfmBuilder::new().with_wrap_hash(super::edsl::WrapHash::production());
    let (arenas, closed) = emit_both_legs(&mut b, &h);
    let program = compile(b.finish());
    execute(&program, &arenas, &crate::hash_pin::BLOCK_HASHER)
        .expect("the uneven one-row schedule verifies");
    assert_eq!(permutations(&program), closed);
}
