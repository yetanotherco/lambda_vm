//! A node verifies a Poseidon1 LFM child proof in-guest (D-P3B P2): the
//! trivial program proved under `hash_pin::P1Family`, harvested into the
//! node's arenas, and one leg (`per_table_aggregator::emit_leg`) emitted under
//! `WrapHash::Poseidon1` — ZisK's transcript replay with the family's
//! statement tag, the width-8 grind check, the tagged leaf hash and the 4-ary
//! child-order walks to each tree's C2 cap. The honest arenas execute; every
//! tamper below is refused by the machine, never by the host harvest.

use stark::proof::options::{GoldilocksCubicProofOptions, ProofOptions};

use crate::hash_pin::{P1Family, ProofFamily};
use crate::tables::types::FE;

use super::builder::LfmBuilder;
use super::compiler::{LfmProgram, compile};
use super::executor::execute;
use super::harvest::{HarvestedChild, harvest_child_under, try_child_arena_words};
use super::hash::HasherKind;
use super::per_table_aggregator::{DerivedChild, declare_leg_arenas, emit_leg};
use super::programs::trivial_program;
use super::proof::{LfmProof, lfm_prove_under, verify_against_artifacts_under};
use super::registry::{LfmArtifacts, build_artifacts_under};
use super::word::LfmWord;

/// The child's options: the production format (cap auto — which an arity-4
/// configuration replaces by its own C2 — and `fri = dp`) at blowup 4, with
/// few queries so the node stays small, and a grind so the leg checks one.
fn options() -> ProofOptions {
    let mut o = GoldilocksCubicProofOptions::with_blowup(4).expect("options");
    o.fri_number_of_queries = 6;
    o.grinding_factor = 4;
    o.fri_final_poly_log_degree = 2;
    crate::zf_format::ZfFormat::DEFAULT.options(o)
}

fn arenas() -> Vec<Vec<LfmWord>> {
    vec![
        (0..4u64)
            .map(|i| core::array::from_fn(|j| FE::from(1_000 * (i + 1) + j as u64)))
            .collect(),
    ]
}

struct P1Child {
    artifacts: LfmArtifacts,
    proof: LfmProof,
    opts: ProofOptions,
}

fn p1_child() -> P1Child {
    let opts = options();
    let program = trivial_program();
    let hasher = super::registry::build_artifacts(&program, &opts).hasher;
    let artifacts =
        build_artifacts_under::<<P1Family as ProofFamily>::Stark>(&program, &opts, hasher);
    let proof = lfm_prove_under::<P1Family>(&program, &artifacts, &arenas(), &opts).expect("prove");
    assert!(verify_against_artifacts_under::<P1Family>(
        &artifacts,
        &proof.proof,
        &proof.public_words,
        &opts
    ));
    P1Child {
        artifacts,
        proof,
        opts,
    }
}

fn harvest(c: &P1Child) -> HarvestedChild {
    harvest_child_under::<P1Family>(c.artifacts.clone(), c.opts.clone(), &c.proof)
        .unwrap_or_else(|e| panic!("{e}"))
}

/// One leg over the child, under `wrap`, with the child's statement tag
/// `tag` (the family's, unless a mutation drops it).
fn node(c: &P1Child, tag: Option<&'static [u8]>) -> LfmProgram {
    let derived = DerivedChild::from_artifacts_under::<P1Family>(
        &c.artifacts,
        &c.opts,
        c.proof.public_words.len(),
    )
    .unwrap_or_else(|e| panic!("{e}"));
    let mut shape = derived.shape();
    shape.statement_tag = tag;
    let mut b = LfmBuilder::new().with_wrap_hash(P1Family::WRAP);
    let a = declare_leg_arenas(&mut b, &shape);
    let _ = emit_leg(&mut b, &shape, &a);
    compile(b.finish())
}

fn runs(program: &LfmProgram, child: &HarvestedChild) -> bool {
    let arenas = try_child_arena_words(child).unwrap_or_else(|e| panic!("{e}"));
    execute(program, &arenas, &HasherKind::Poseidon1W16).is_ok()
}

fn flip(c: &mut [u8; 32]) {
    c[31] ^= 1;
}

#[test]
fn a_p1_node_verifies_a_p1_child_and_refuses_its_tampers() {
    let c = p1_child();
    let honest = harvest(&c);
    let program = node(&c, P1Family::STATEMENT_TAG);
    assert!(
        program.hasher(crate::hash_pin::BLOCK_HASHER) == HasherKind::Poseidon1W16,
        "a P1 node's hash rows are the width-16 socket's"
    );
    assert!(runs(&program, &honest), "the honest child executes");
    // The P1 child's trees are capped at C2: a cap is hinted per tree.
    assert!(
        honest.legs.iter().any(|l| !l.caps.is_empty()),
        "some table carries caps"
    );

    let refused = |what: &str, edit: &dyn Fn(&mut HarvestedChild)| {
        let mut bad = harvest(&c);
        edit(&mut bad);
        assert!(!runs(&program, &bad), "{what}: the node must refuse");
    };
    // A table whose trees are capped and whose FRI commits a layer.
    let t = honest
        .legs
        .iter()
        .position(|l| !l.caps.is_empty() && !l.fri_openings[0].is_empty())
        .expect("a capped table with a committed FRI layer");
    refused("main root", &|h| flip(&mut h.tables[t].main_root));
    refused("composition root", &|h| {
        flip(&mut h.tables[t].composition_root)
    });
    refused("trace sibling", &|h| {
        flip(&mut h.legs[t].openings[1][0].1[0])
    });
    refused("cap node", &|h| {
        let last = h.legs[t].caps.len() - 1;
        flip(&mut h.legs[t].caps[last])
    });
    refused("FRI sibling", &|h| {
        flip(&mut h.legs[t].fri_openings[2][0].1[0])
    });
    refused("FRI root", &|h| flip(&mut h.tables[t].fri_roots[0]));
    refused("opened value", &|h| {
        h.legs[t].openings[0][0].0[0][0] += FE::from(1u64)
    });
    refused("OOD value (the transcript)", &|h| {
        h.tables[t].ood_current[0] += crate::tables::types::FEE::from(1u64)
    });
    refused("nonce", &|h| {
        h.tables[t].nonce = h.tables[t].nonce.map(|n| n ^ 1)
    });
    // ★ The query index is the transcript's: swapping two queries' openings
    // (each internally consistent) moves every leaf to another index.
    refused("swapped queries", &|h| {
        h.legs[t].openings.swap(0, 1);
        h.legs[t].fri_openings.swap(0, 1);
    });
    refused("a claimed word", &|h| {
        h.public_words[0].1[0] += FE::from(1u64)
    });
}

/// The family's statement tag is load-bearing in the guest: a node that
/// replays the child's statement without it refuses the honest child.
#[test]
fn a_node_without_the_family_tag_refuses_the_p1_child() {
    let c = p1_child();
    let honest = harvest(&c);
    assert!(runs(&node(&c, P1Family::STATEMENT_TAG), &honest));
    assert!(!runs(&node(&c, None), &honest));
}
