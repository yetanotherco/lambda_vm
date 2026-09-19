//! Gates for the continuation statements.
//!
//! The host functions are `multilinear_continuation::absorb_epoch` and
//! `absorb_global`, both `pub(crate)` and both callable from here, so the gate
//! is the same function and not a model of it.
//!
//! A transcript's STATE is not observable; its next draw is. So every value gate
//! here drives the host through the real `absorb_epoch`, then the 32-byte root a
//! verifier absorbs next (`multilinear_table.rs:659`), then
//! `sample_field_element` — and drives the machine through the same three steps,
//! and compares the CHALLENGE. One byte wrong anywhere in the statement moves it.

use crypto::fiat_shamir::default_transcript::DefaultTranscript;
use crypto::fiat_shamir::is_transcript::IsTranscript;
use crypto::fiat_shamir::transcript_hash::RpxTranscriptHash;
use multilinear::whir_chain::{ChainConfig, GrindBits};

use crate::TableCounts;
use crate::statement::statement_padding;
use crate::tables::types::{FE, FEE, GoldilocksExtension};

use super::whir_transcript::DIGEST_FELTS;

use super::algebraic_commit::commitment_to_digest;
use super::builder::LfmBuilder;
use super::compiler::compile;
use super::executor::execute;
use super::validator::validate;
use super::whir_statement::{
    EpochStatement, GlobalStatement, StatementCost, emit_epoch_statement, emit_global_statement,
    epoch_statement_bytes, global_statement_bytes, statement_cost,
};
use super::whir_transcript::WhirTranscript;
use super::word::{LfmWord, base_word, word_as_ext};

type E = GoldilocksExtension;
type HostTranscript = DefaultTranscript<E, RpxTranscriptHash>;

/// The posture the whole VM proof runs at (`multilinear_prove.rs:93`), so the
/// config words and the grind trailer are the block's own and not invented.
fn config() -> ChainConfig {
    ChainConfig::with_security(2, 4, 25, 128, GrindBits::uniform(20))
}

fn digest(seed: u8) -> [u8; 32] {
    core::array::from_fn(|i| seed.wrapping_add(i as u8).wrapping_mul(37))
}

/// A `TableCounts` with every field distinct, so a reordered absorb is a
/// different stream.
fn counts() -> TableCounts {
    TableCounts {
        cpu: 3,
        lt: 5,
        memw: 7,
        memw_aligned: 11,
        load: 13,
        mul: 17,
        dvrm: 19,
        shift: 23,
        branch: 29,
        memw_register: 31,
        eq: 37,
        bytewise: 41,
        store: 43,
        cpu32: 47,
        keccak: 1,
        keccak_rnd: 1,
        ecsm: 1,
        ecdas: 1,
        hint: 1,
        commit: 1,
        blake3: 1,
    }
}

/// A root, as the verifier absorbs it: 32 bytes, and the same word the machine
/// hints.
fn root(seed: u8) -> ([u8; 32], LfmWord) {
    let bytes = digest(seed);
    (bytes, commitment_to_digest(&bytes))
}

/// The host's challenge after `absorb_epoch`, one root, and a draw.
fn host_epoch_challenge(
    elf: &[u8; 32],
    label: u64,
    public_output: &[u8],
    table_counts: &TableCounts,
    table_num_vars: &[u8],
    root_bytes: &[u8; 32],
) -> FEE {
    let mut transcript = HostTranscript::new(&[]);
    crate::multilinear_continuation::absorb_epoch(
        &mut transcript,
        elf,
        public_output,
        table_counts,
        label,
        table_num_vars,
        &config(),
    );
    transcript.append_bytes(root_bytes);
    transcript.sample_field_element()
}

/// The machine's challenge over the same three steps, and what the statement
/// cost while doing it.
fn machine_epoch_challenge(
    elf: &[u8; 32],
    label: u64,
    public_output: &[u8],
    table_counts: &TableCounts,
    table_num_vars: &[u8],
    root_word: LfmWord,
) -> (FEE, StatementCost, usize, usize) {
    let mut b = LfmBuilder::new().with_wrap_hash(super::edsl::WrapHash::production());
    let arena = b.declare_arena(1);
    let mut transcript = WhirTranscript::new();

    let cost = emit_epoch_statement(
        &mut transcript,
        &EpochStatement {
            elf_digest: elf,
            epoch_label: label,
            public_output,
            table_counts,
            table_num_vars,
            config: &config(),
        },
    );

    let root = b.hint_word(arena, 0);
    transcript.absorb_digest(&mut b, root);
    let challenge = transcript.sample_ext(&mut b);
    b.public(challenge.as_cell());

    let program = compile(b.finish());
    validate(&program).expect("the statement leg must be admissible");
    let rows = program.instrs.len();
    let consts = super::whir_chain_tests::const_rows(&program);
    let exec = execute(&program, &[vec![root_word]], &crate::hash_pin::BLOCK_HASHER)
        .expect("the statement leg executes");
    let drawn = word_as_ext(&exec.public_words[0].1).expect("a published challenge");
    (drawn, cost, rows, consts)
}

/// The shapes: the three the block measures, plus every residue of the two
/// variable lengths so the pad is exercised at all eight.
fn block_shapes() -> Vec<(usize, usize, usize)> {
    // (|public_output|, |table_num_vars|, the pad measured on the block)
    vec![(0, 34, 1), (0, 33, 2), (160, 32, 3)]
}

/// ★ GATE ONE: the machine's challenge is the host's, on the block's own shapes.
#[test]
fn the_epoch_statement_draws_the_challenge_the_host_draws() {
    let elf = digest(0x11);
    let table_counts = counts();
    for (output_len, num_vars_len, _) in block_shapes() {
        let public_output: Vec<u8> = (0..output_len)
            .map(|i| (i as u8).wrapping_mul(13))
            .collect();
        let table_num_vars: Vec<u8> = (0..num_vars_len).map(|i| 12 + (i as u8) % 9).collect();
        let (root_bytes, root_word) = root(0x5a);

        let want = host_epoch_challenge(
            &elf,
            7,
            &public_output,
            &table_counts,
            &table_num_vars,
            &root_bytes,
        );
        let (got, cost, rows, consts) = machine_epoch_challenge(
            &elf,
            7,
            &public_output,
            &table_counts,
            &table_num_vars,
            root_word,
        );

        println!(
            "epoch statement |po|={output_len} |tnv|={num_vars_len}: len={} pad={} felts={} \
             constants={} ({consts} in the program, {rows} instructions)",
            cost.len,
            cost.pad,
            cost.felts,
            cost.constants.len(),
        );
        assert_eq!(
            got, want,
            "|po|={output_len} |tnv|={num_vars_len}: the machine must draw the host's challenge"
        );
    }
}

/// ★ GATE TWO: the pads the block measured, reproduced from the byte stream.
///
/// ⚠ The three epoch values are the `WHIR-PAD` lines from the a2r/a2u runs; they
/// are what the formula has to reproduce, not what it is fitted to. The formula
/// itself — `(3 − |po| − |tnv|) mod 8` — is asserted beside the stream so that a
/// field added to `absorb_epoch` breaks the arithmetic rather than silently
/// moving both.
#[test]
fn the_epoch_pad_is_the_one_the_block_measured() {
    let elf = digest(0x22);
    let table_counts = counts();
    for (output_len, num_vars_len, measured) in block_shapes() {
        let public_output = vec![0u8; output_len];
        let table_num_vars = vec![9u8; num_vars_len];
        let bytes = epoch_statement_bytes(&EpochStatement {
            elf_digest: &elf,
            epoch_label: 0,
            public_output: &public_output,
            table_counts: &table_counts,
            table_num_vars: &table_num_vars,
            config: &config(),
        });
        let cost = statement_cost(&bytes);

        println!(
            "epoch pad |po|={output_len} |tnv|={num_vars_len}: len={} pad={} (block measured \
             {measured})",
            cost.len, cost.pad
        );
        // ★ DERIVED, not a literal. The fixed part is tag 42 + digest 32 +
        // label 8 + |po| prefix 8 + counts 8·NUM_TABLE_KINDS + |tnv| prefix 8 +
        // config 24 + trailer 3. That was 245 at fifteen kinds and is 293 at
        // twenty-one, and the main-sync port is what showed the cost of writing
        // it out: a literal turns an encoding change into a red test HERE
        // instead of naming the encoding that moved.
        let fixed = 42 + 32 + 8 + 8 + 8 * crate::statement::NUM_TABLE_KINDS + 8 + 24 + 3;
        assert_eq!(
            cost.len,
            fixed + output_len + num_vars_len,
            "the fixed part is {fixed} bytes at {} table kinds: tag 42 + digest 32 + label 8 \
             + 8 + counts 8·kinds + 8 + config 24 + trailer 3",
            crate::statement::NUM_TABLE_KINDS
        );
        assert_eq!(
            cost.pad,
            (3 + 16 - (output_len % 8) - (num_vars_len % 8)) % 8,
            "epoch pad = (3 - |po| - |tnv|) mod 8"
        );
        assert_eq!(cost.pad, measured, "the pad the block measured");
        assert_eq!(
            (cost.len + cost.pad) % 8,
            0,
            "the statement ends on a felt boundary"
        );
    }
}

/// ★ The GLOBAL pad, and why the recorded formula is the same one.
///
/// ⚠ I wrote that `global pad = (2 − epochs − pages) mod 8` was refuted by the
/// measurement, and this test refuted ME: it gives 0 at the block's shape, which
/// IS the measured pad. The two forms are identical because `table_num_vars` is
/// one byte per table and the cross-epoch proof's tables are every epoch's
/// bookend plus the global-memory tables (`multilinear_continuation.rs:556`), so
/// `|table_num_vars| = epochs + pages`. The identity is asserted below, because
/// it is what makes the recorded form correct — and what would stop being true
/// if the global's table set ever changed shape.
#[test]
fn the_global_pad_is_the_one_the_block_measured() {
    let elf = digest(0x33);
    let pages: Vec<u64> = (0..35u64).map(|i| i * 4096).collect();
    let table_num_vars = vec![14u8; 50];
    let bytes = global_statement_bytes(&GlobalStatement {
        elf_digest: &elf,
        num_epochs: 15,
        num_private_input_pages: 0,
        page_bases: &pages,
        table_num_vars: &table_num_vars,
        config: &config(),
    });
    let cost = statement_cost(&bytes);

    println!(
        "global pad page_bases={} table_num_vars={}: len={} pad={} felts={}",
        pages.len(),
        table_num_vars.len(),
        cost.len,
        cost.pad,
        cost.felts
    );
    assert_eq!(
        cost.len,
        134 + 8 * pages.len() + table_num_vars.len(),
        "the fixed part is 134 bytes: tag 43 + digest 32 + epochs 8 + pages 8 + 8 + 8 + \
         config 24 + trailer 3"
    );
    assert_eq!(
        cost.pad,
        (2 + 8 - table_num_vars.len() % 8) % 8,
        "global pad = (2 - |table_num_vars|) mod 8"
    );
    assert_eq!(cost.pad, 0, "the pad the block measured at this shape");

    let epochs = 15usize;
    assert_eq!(
        table_num_vars.len(),
        epochs + pages.len(),
        "one byte per table: every epoch's bookend plus one global-memory table per page — \
         the identity the recorded formula rests on"
    );
    let recorded = (2 + 64 - epochs - pages.len()) % 8;
    assert_eq!(
        recorded, cost.pad,
        "the recorded `(2 - epochs - pages) mod 8` is the byte stream's own pad, through that \
         identity and not by coincidence"
    );
}

/// ⛔ THE GLOBAL PAD IDENTITY, SWEPT — because its sibling above pins it at ONE
/// shape, and at that shape the pad is ZERO.
///
/// `the_global_pad_is_the_one_the_block_measured` asserts
/// `pad = (2 − epochs − pages) mod 8` against the byte stream at the block's
/// fifteen epochs and thirty-five pages. Both sides are 0 there. A form that was
/// wrong by a multiple of eight — or wrong in a way that happens to vanish at
/// that one residue — would agree with the stream and the test would be green:
/// the quantity it reads cannot move under a whole class of the errors it exists
/// to catch.
///
/// So this sweeps the shape and asserts two things the single-shape arm cannot:
/// that the identity holds at EVERY residue, and that the sweep actually
/// REACHES the nonzero ones. Without that second assertion a sweep that happened
/// to visit only pad-0 shapes would be the same vacuous check with more
/// iterations.
#[test]
fn the_global_pad_identity_holds_at_every_residue() {
    let elf = digest(0x35);
    let mut seen = [false; 8];
    for epochs in 1usize..=16 {
        for pages in 0usize..=16 {
            let page_bases: Vec<u64> = (0..pages as u64).map(|i| i * 4096).collect();
            let table_num_vars: Vec<u8> = vec![14u8; epochs + pages];
            let bytes = global_statement_bytes(&GlobalStatement {
                elf_digest: &elf,
                num_epochs: epochs as u64,
                num_private_input_pages: 0,
                page_bases: &page_bases,
                table_num_vars: &table_num_vars,
                config: &config(),
            });
            let cost = statement_cost(&bytes);
            // The stream's own pad, measured.
            assert_eq!(
                cost.len,
                134 + 8 * pages + table_num_vars.len(),
                "the fixed part is 134 bytes at every shape"
            );
            // ★ The RECORDED form, evaluated at this shape. `page_bases` is
            // eight bytes an entry and cannot move the alignment, which is why
            // only the table count appears in it.
            let recorded = (2 + 64 - epochs - pages) % 8;
            assert_eq!(
                recorded, cost.pad,
                "the recorded `(2 - epochs - pages) mod 8` missed the stream's pad at \
                 {epochs} epochs and {pages} pages: form {recorded}, stream {}",
                cost.pad,
            );
            seen[cost.pad] = true;
        }
    }
    println!(
        "global pad residues reached by the sweep: {:?}",
        (0..8).filter(|r| seen[*r]).collect::<Vec<_>>()
    );
    // ⛔ THE ANTI-VACUITY ASSERT. A sweep that only ever saw pad 0 would agree
    // with any form that is right at zero, which is the defect the single-shape
    // arm has and the reason this one exists.
    assert!(
        seen.iter().all(|reached| *reached),
        "the sweep must reach every residue, or the identity is pinned only where it \
         happens to vanish"
    );
}

/// ★ GATE THREE: the row form. A statement costs its interned constants and no
/// operation at all.
#[test]
fn the_epoch_statement_emits_only_its_constants() {
    let elf = digest(0x44);
    let table_counts = counts();
    for (output_len, num_vars_len, _) in block_shapes() {
        let public_output: Vec<u8> = (0..output_len).map(|i| (i as u8).wrapping_mul(7)).collect();
        let table_num_vars: Vec<u8> = (0..num_vars_len).map(|i| 10 + (i as u8) % 7).collect();
        let (_, root_word) = root(0x6b);
        let (_, cost, rows, consts) = machine_epoch_challenge(
            &elf,
            3,
            &public_output,
            &table_counts,
            &table_num_vars,
            root_word,
        );

        // The program is: one Hint for the root, one Unpack for the digest, one
        // Pack for the draw, one Public — plus the statement's constants and the
        // squeeze's own rows.
        let squeeze_rows = super::whir_transcript::squeeze_rows(cost.felts + DIGEST_FELTS);
        // One `Hint` for the root, the `Unpack` its absorb spends, the `Pack`
        // the draw spends, and the published challenge.
        let plumbing = 1
            + super::whir_transcript::absorb_unpack_rows()
            + super::whir_transcript::sample_ext_rows()
            + 1;
        println!(
            "epoch statement rows |po|={output_len} |tnv|={num_vars_len}: {rows} instructions \
             = {consts} constants + {squeeze_rows} squeeze + {plumbing} plumbing; the form \
             says {} statement constants unioned with the squeeze's two, and {} operations",
            cost.constants.len(),
            cost.operations(),
        );
        // ★ The program interns more than the statement does, and the extras are
        // the SQUEEZE's, named rather than absorbed into a fudge (instance 63).
        // `algebraic_leaf_hash` (`edsl.rs:676-692`) interns two things: the
        // `leaf_capacity` of the leaf it is about to hash — once per distinct
        // leaf LENGTH per program, which no per-hash row form can see — and the
        // ZERO it pads a partial word with, which doubles as its empty digest.
        // ⚠ The zero is free whenever the statement already has a zero group and
        // costs a row when it does not, which is why this is a union by VALUE
        // and not a `+ 2`: at |po| = 0 the statement supplies the zero and the
        // program interns one extra; at |po| = 160 it does not and the program
        // interns two.
        let mut predicted = cost.constants.clone();
        for word in [
            super::algebraic_commit::leaf_capacity(cost.felts + DIGEST_FELTS),
            base_word(FE::zero()),
        ] {
            if !predicted.contains(&word) {
                predicted.push(word);
            }
        }
        assert_eq!(
            consts,
            predicted.len(),
            "|po|={output_len} |tnv|={num_vars_len}: the statement's constants, unioned with \
             the leaf capacity and the zero its squeeze interns"
        );
        assert_eq!(
            rows,
            consts + squeeze_rows + plumbing,
            "|po|={output_len} |tnv|={num_vars_len}: the statement adds no operation of its own"
        );
        assert_eq!(cost.operations(), 0, "a statement emits no operation row");
    }
}

/// ★ GATE FOUR: the refusal does not fire after a pad — and DOES fire without
/// one.
///
/// `absorb_felts` refuses a runtime absorb that does not start on a felt
/// boundary, because a straddling value would need a byte shift. The pad is what
/// makes it not fire. Both halves are here: without the second, "it does not
/// fire" is a check that cannot fail.
#[test]
fn the_pad_is_what_keeps_the_roots_aligned() {
    let elf = digest(0x55);
    let table_counts = counts();
    let mut padded = 0usize;
    for output_len in 0..8usize {
        for num_vars_len in 0..8usize {
            let public_output = vec![1u8; output_len];
            let table_num_vars = vec![2u8; num_vars_len];
            let statement = EpochStatement {
                elf_digest: &elf,
                epoch_label: 1,
                public_output: &public_output,
                table_counts: &table_counts,
                table_num_vars: &table_num_vars,
                config: &config(),
            };
            let bytes = epoch_statement_bytes(&statement);
            let pad = statement_padding(bytes.len());
            if pad > 0 {
                padded += 1;
            }

            // With the pad: the root absorbs.
            let mut b = LfmBuilder::new().with_wrap_hash(super::edsl::WrapHash::production());
            let arena = b.declare_arena(1);
            let mut transcript = WhirTranscript::new();
            emit_epoch_statement(&mut transcript, &statement);
            let root = b.hint_word(arena, 0);
            transcript.absorb_digest(&mut b, root);

            // Without it: the same absorb has no felt boundary to start on, and
            // the transcript must refuse rather than shift.
            let mut b2 = LfmBuilder::new().with_wrap_hash(super::edsl::WrapHash::production());
            let arena2 = b2.declare_arena(1);
            let mut bare = WhirTranscript::new();
            bare.absorb_const_bytes(&bytes);
            let root2 = b2.hint_word(arena2, 0);
            let refused = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                bare.absorb_digest(&mut b2, root2);
            }));
            assert_eq!(
                refused.is_err(),
                pad > 0,
                "|po|={output_len} |tnv|={num_vars_len}: an unpadded statement of {} bytes must \
                 refuse a runtime absorb exactly when it does not already end on a boundary",
                bytes.len()
            );
        }
    }
    assert_eq!(
        padded, 56,
        "seven of every eight residue pairs need a pad — otherwise the control half above \
         never runs"
    );
}

/// ★ The sponge entry the statement hands on, which is what an epoch program
/// threads into its first table.
#[test]
fn the_statement_hands_on_the_sponge_it_leaves() {
    let elf = digest(0x66);
    let table_counts = counts();
    let public_output = vec![0u8; 160];
    let table_num_vars = vec![13u8; 32];
    let bytes = epoch_statement_bytes(&EpochStatement {
        elf_digest: &elf,
        epoch_label: 2,
        public_output: &public_output,
        table_counts: &table_counts,
        table_num_vars: &table_num_vars,
        config: &config(),
    });
    let cost = statement_cost(&bytes);
    let entry = cost.entry();

    println!(
        "entry after the statement: buffered_felts={} out_pos={}",
        entry.buffered_felts, entry.out_pos
    );
    assert_eq!(
        entry.buffered_felts, cost.felts,
        "the sponge holds the statement's felts, unhashed"
    );
    assert_eq!(
        entry.out_pos,
        super::whir_transcript::CANDIDATES_PER_SQUEEZE,
        "every absorb invalidates the output buffer, so nothing is in hand"
    );
    assert_eq!(cost.felts, (cost.len + cost.pad) / 8, "felts are bytes / 8");
}

/// A zero group is interned once however many times the statement repeats it —
/// the pool is keyed on the value.
#[test]
fn the_statements_constants_are_distinct_values() {
    let elf = digest(0x77);
    let table_counts = counts();
    let public_output = vec![0u8; 64];
    let table_num_vars = vec![0u8; 40];
    let bytes = epoch_statement_bytes(&EpochStatement {
        elf_digest: &elf,
        epoch_label: 0,
        public_output: &public_output,
        table_counts: &table_counts,
        table_num_vars: &table_num_vars,
        config: &config(),
    });
    let cost = statement_cost(&bytes);
    let groups = (cost.len + cost.pad) / 8;
    println!(
        "constants: {} distinct of {groups} groups",
        cost.constants.len()
    );
    assert!(
        cost.constants.len() < groups,
        "a statement this full of zeroes must share rows between them"
    );
    let zero = base_word(FE::zero());
    assert!(
        cost.constants.contains(&zero),
        "the zero group is one of them"
    );
    let mut seen: Vec<LfmWord> = Vec::new();
    for word in &cost.constants {
        assert!(!seen.contains(word), "the pool holds no duplicate");
        seen.push(*word);
    }
}

/// ★ The GLOBAL statement's challenge is the host's too.
///
/// Item 6 builds the cross-epoch program on this stream; gating it here, beside
/// the epoch's, costs one fixture and means item 6 starts from a pinned
/// statement rather than from `absorb_global`'s source.
#[test]
fn the_global_statement_draws_the_challenge_the_host_draws() {
    let elf = digest(0x88);
    let pages: Vec<u64> = (0..35u64).map(|i| 0x1000 + i * 4096).collect();
    let table_num_vars: Vec<u8> = (0..50).map(|i| 11 + (i as u8) % 5).collect();
    let (root_bytes, root_word) = root(0x9c);

    let mut host = HostTranscript::new(&[]);
    crate::multilinear_continuation::absorb_global(
        &mut host,
        &elf,
        15,
        0,
        &pages,
        &table_num_vars,
        &config(),
    );
    host.append_bytes(&root_bytes);
    let want = host.sample_field_element();

    let mut b = LfmBuilder::new().with_wrap_hash(super::edsl::WrapHash::production());
    let arena = b.declare_arena(1);
    let mut transcript = WhirTranscript::new();
    let cost = emit_global_statement(
        &mut transcript,
        &GlobalStatement {
            elf_digest: &elf,
            num_epochs: 15,
            num_private_input_pages: 0,
            page_bases: &pages,
            table_num_vars: &table_num_vars,
            config: &config(),
        },
    );
    let root_cell = b.hint_word(arena, 0);
    transcript.absorb_digest(&mut b, root_cell);
    let challenge = transcript.sample_ext(&mut b);
    b.public(challenge.as_cell());
    let program = compile(b.finish());
    validate(&program).expect("the global statement leg must be admissible");
    let exec = execute(&program, &[vec![root_word]], &crate::hash_pin::BLOCK_HASHER)
        .expect("the global statement leg executes");
    let got = word_as_ext(&exec.public_words[0].1).expect("a published challenge");

    println!(
        "global statement: len={} pad={} felts={} constants={}",
        cost.len,
        cost.pad,
        cost.felts,
        cost.constants.len()
    );
    assert_eq!(
        got, want,
        "the machine must draw the challenge the host draws after `absorb_global`"
    );
    assert_eq!(cost.operations(), 0, "a statement emits no operation row");
}
