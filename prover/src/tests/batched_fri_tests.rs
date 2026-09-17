//! The batched FRI, pinned at the one size where it has to agree with the
//! unbatched one.

use crate::tables::MaxRowsConfig;
use crate::tables::types::{GoldilocksExtension, GoldilocksField};
use executor::elf::Elf;
use stark::prover::IsStarkProver;
use stark::prover::TableDeep;

/// A batch of one must reproduce the proof's FRI exactly.
///
/// `Σ αᵏ·deepₖ` over a single member is `deepₖ`, so the batched path and the
/// per-table one fold the same codeword over the same domain. If their layer
/// roots differ, the difference is in the codeword or in the domain — which is
/// the whole substance of the batching, and worth catching before any group has
/// more than one member in it.
#[test]
fn a_batch_of_one_matches_the_unbatched_fri() {
    type P = stark::prover::Prover<GoldilocksField, GoldilocksExtension, ()>;

    let elf_bytes = crate::test_utils::asm_elf_bytes("fib_iterative_160k");
    let elf = Elf::load(&elf_bytes).expect("ELF load");
    let max_rows = MaxRowsConfig {
        cpu: 1 << 15,
        memw: 1 << 10,
        load: 1 << 10,
        branch: 1 << 12,
        ..Default::default()
    };
    let proof_options = stark::proof::options::GoldilocksCubicProofOptions::with_blowup(2)
        .expect("blowup 2 is valid");

    let vm_proof = crate::prove_with_options_and_inputs(&elf_bytes, &[], &proof_options, &max_rows)
        .expect("ordinary prove");

    let committed = crate::commit_phase::run_to_end(&elf, &[], &max_rows, &proof_options)
        .expect("commit phase");
    let challenge = crate::challenge_phase::run(&committed, &elf, &elf_bytes, &proof_options)
        .expect("challenge phase");
    drop(committed);

    let page_configs = crate::tables::trace_builder::Traces::page_configs_from_elf_and_runtime(
        &elf,
        &vm_proof.runtime_page_ranges,
        vm_proof.num_private_input_pages,
        vm_proof.proof.proofs.len(),
    )
    .expect("page configs");
    let airs = crate::VmAirs::new(
        &elf,
        &proof_options,
        false,
        &page_configs,
        &vm_proof.table_counts,
        None,
        true,
        None,
        None,
        None,
    );
    let mut resident = crate::logup_phase::resident_tables(&elf, &[], &max_rows).expect("resident");

    // BITWISE is table 0 and the largest resident one, so it exercises a real
    // domain rather than a one-row corner.
    let idx = 0usize;
    let n = challenge.roots.len();
    let mut fork = crate::logup_phase::fork_for(&challenge, idx, n);
    let deep = <P as IsStarkProver<_, _, _>>::deep_for_table(
        airs.bitwise.as_ref(),
        &(),
        &mut resident.bitwise,
        &challenge.challenges,
        &mut fork,
    )
    .expect("deep");

    let one = math::field::element::FieldElement::<GoldilocksExtension>::one();
    let fri = <P as IsStarkProver<_, _, _>>::batch_fri(
        airs.bitwise.as_ref(),
        vec![deep],
        &one,
        &mut fork,
    )
    .expect("batched fri");

    // Everything round 4 would have produced for this table on its own: the
    // layers, the final polynomial, the ground nonce, and the queries its
    // openings answer.
    let want = &vm_proof.proof.proofs[idx];
    assert_eq!(
        fri.layer_roots, want.fri_layers_merkle_roots,
        "a batch of one folded to a different FRI than the proof carries"
    );
    assert_eq!(
        fri.final_poly_coeffs, want.fri_final_poly_coeffs,
        "a batch of one folded to a different final polynomial"
    );
    // And there the comparison stops. Grinding searches for a nonce in parallel
    // and finds whichever one it finds first, so two runs over the same
    // transcript state produce different valid nonces — and the queries are
    // sampled after the nonce is absorbed, so they differ with it. What is
    // deterministic is everything up to that point, which is what is checked
    // above; the queries are checked instead by the count they produce.
    assert_eq!(
        fri.query_list.len(),
        want.query_list.len(),
        "a batch of one answered a different number of queries"
    );
    assert_eq!(
        fri.iotas.len(),
        want.query_list.len(),
        "the group sampled a different number of query indices than it decommitted"
    );
}

/// The batch's coefficient must depend on every table it folds.
///
/// `alpha` is what makes a batched fold binding: if it could be drawn without
/// some table's round-3 data, that table could be swapped after the coefficient
/// was fixed and the fold would still check out. So the property to pin is not
/// that the derivation runs — it is that moving any single field any table
/// contributes moves the result.
///
/// Built from data rather than from a proof on purpose: this is about the byte
/// order, and a synthetic table exercises every field including the ones a real
/// fixture might leave empty.
#[test]
fn alpha_moves_when_any_table_moves() {
    use crypto::fiat_shamir::default_transcript::DefaultTranscript;
    use math::field::element::FieldElement;
    use stark::table::Table;
    type P = stark::prover::Prover<GoldilocksField, GoldilocksExtension, ()>;
    type E = GoldilocksExtension;

    let table = |seed: u64| TableDeep::<E> {
        lde_size: 8,
        trace_rows: 4,
        deep: Vec::new(),
        bus_contribution: Some(FieldElement::<E>::from(seed)),
        composition_poly_root: [seed as u8; 32],
        trace_ood: Table::new(vec![FieldElement::<E>::from(seed + 1)], 1),
        trace_ood_next: Table::new(vec![FieldElement::<E>::from(seed + 2)], 1),
        parts_ood: vec![FieldElement::<E>::from(seed + 3)],
    };

    let pre_fork = DefaultTranscript::<E>::new(&[7, 7, 7]);
    let base = vec![table(1), table(2)];
    let alpha = <P as IsStarkProver<_, _, _>>::batch_alpha(&pre_fork, &base);

    // Every field of every table, one at a time.
    type Mutation = (&'static str, Box<dyn Fn(&mut Vec<TableDeep<E>>)>);
    let mutate: Vec<Mutation> = vec![
        (
            "bus",
            Box::new(|t: &mut Vec<TableDeep<E>>| {
                t[0].bus_contribution = Some(FieldElement::<E>::from(99))
            }),
        ),
        (
            "root",
            Box::new(|t: &mut Vec<TableDeep<E>>| t[1].composition_poly_root = [9u8; 32]),
        ),
        (
            "ood",
            Box::new(|t: &mut Vec<TableDeep<E>>| {
                t[0].trace_ood = Table::new(vec![FieldElement::<E>::from(99)], 1)
            }),
        ),
        (
            "ood_next",
            Box::new(|t: &mut Vec<TableDeep<E>>| {
                t[1].trace_ood_next = Table::new(vec![FieldElement::<E>::from(99)], 1)
            }),
        ),
        (
            "parts",
            Box::new(|t: &mut Vec<TableDeep<E>>| {
                t[0].parts_ood = vec![FieldElement::<E>::from(99)]
            }),
        ),
    ];
    for (what, f) in mutate {
        let mut moved = base.clone();
        f(&mut moved);
        assert_ne!(
            alpha,
            <P as IsStarkProver<_, _, _>>::batch_alpha(&pre_fork, &moved),
            "alpha ignores {what}, so that data is not bound to the fold"
        );
    }

    // And order is part of it: the same tables the other way round are a
    // different batch.
    let swapped = vec![base[1].clone(), base[0].clone()];
    assert_ne!(
        alpha,
        <P as IsStarkProver<_, _, _>>::batch_alpha(&pre_fork, &swapped),
        "alpha ignores the table order, which the verifier replays"
    );
}

/// The driver must fold every table, and fold them by domain.
///
/// Two things can silently go wrong at once here. A table can go missing — the
/// walk produces the chunked ones and the end-of-run step the rest, and a batch
/// that skips one is not a smaller batch, it is a wrong one. And tables of
/// different domains can end up in the same group, which the fold cannot
/// express: it squares the coset offset each layer, so a short codeword never
/// lines up with a tall fold.
///
/// So this checks the count against a real proof's table count, and that every
/// group is one domain with at least one member, and that the collapse actually
/// happened.
#[test]
fn the_driver_folds_every_table_grouped_by_domain() {
    let elf_bytes = crate::test_utils::asm_elf_bytes("fib_iterative_160k");
    let elf = Elf::load(&elf_bytes).expect("ELF load");
    let max_rows = MaxRowsConfig {
        cpu: 1 << 15,
        memw: 1 << 10,
        load: 1 << 10,
        branch: 1 << 12,
        ..Default::default()
    };
    let proof_options = stark::proof::options::GoldilocksCubicProofOptions::with_blowup(2)
        .expect("blowup 2 is valid");

    let vm_proof = crate::prove_with_options_and_inputs(&elf_bytes, &[], &proof_options, &max_rows)
        .expect("ordinary prove");

    let committed = crate::commit_phase::run_to_end(&elf, &[], &max_rows, &proof_options)
        .expect("commit phase");
    let challenge = crate::challenge_phase::run(&committed, &elf, &elf_bytes, &proof_options)
        .expect("challenge phase");
    drop(committed);
    let batched = crate::logup_phase::run_batched(&elf, &[], &max_rows, &proof_options, &challenge)
        .expect("batched phase");

    let folded: usize = batched.members.iter().sum();
    assert_eq!(
        folded,
        vm_proof.proof.proofs.len(),
        "the driver folded {folded} tables but the proof has {}",
        vm_proof.proof.proofs.len()
    );
    assert!(
        batched.groups.len() < folded,
        "{} groups for {folded} tables is no collapse at all",
        batched.groups.len()
    );
    // A group commits layers exactly when there is something to fold. FRI stops
    // at the final polynomial, whose CODEWORD is the blowup times its degree
    // bound — so with blowup 2 and a degree bound of 2^7, a 256-long codeword is
    // already terminal and folds zero times. The short tables (one row blown up
    // to two) are terminal for the same reason. An empty group there is correct,
    // not a group that failed.
    let terminal =
        (1usize << proof_options.fri_final_poly_log_degree) * proof_options.blowup_factor as usize;
    for ((lde_size, fri), count) in batched.groups.iter().zip(batched.members.iter()) {
        assert!(*count > 0, "a group of {lde_size} folded nothing");
        assert_eq!(
            !fri.layer_roots.is_empty(),
            *lde_size > terminal,
            "a group of {lde_size} committed {} layers against a terminal of {terminal}",
            fri.layer_roots.len()
        );
        assert!(
            !fri.iotas.is_empty(),
            "a group of {lde_size} sampled no queries for its members to open at"
        );
    }
    // Domains are distinct: a repeated one would mean two groups that should
    // have been one, which is a fold that did not happen.
    let mut sizes: Vec<usize> = batched.groups.iter().map(|(s, _)| *s).collect();
    let before = sizes.len();
    sizes.dedup();
    assert_eq!(before, sizes.len(), "two groups share a domain");
}
