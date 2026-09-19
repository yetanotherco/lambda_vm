//! WHIR against FRI on the same programs, same machine, CPU only.
//!
//! Ignored: these are measurements, not assertions. Run one with
//!
//! ```text
//! cargo test --release -p lambda-vm-prover --lib \
//!     multilinear_bench_tests::shapes -- --ignored --nocapture
//! ```
//!
//! `RAYON_NUM_THREADS=1` on both sides is the algorithmic comparison — neither
//! prover gets credit for being better parallelised. All cores is the number a
//! user sees. The gap between them is the parallelisation backlog, which for
//! the multilinear path is most of it: only the Merkle build and the NTT are
//! parallel today.

use std::time::Instant;

use executor::elf::Elf;
use executor::vm::execution::Executor;
use multilinear::whir_chain::GrindBits;
use multilinear::whir_hash::KeccakWhir;
use stark::proof::options::GoldilocksCubicProofOptions;

use crate::multilinear_prove;
use crate::tables::MaxRowsConfig;
use crate::tables::trace_builder::Traces;

/// Blowup 4, 128 bits, 20 bits of grinding — the parameters the multilinear
/// path derives its own from, so the two are being asked for the same security.
pub(super) fn options() -> stark::proof::options::ProofOptions {
    GoldilocksCubicProofOptions::with_params(4, 128, 20).expect("valid options")
}

/// `pub(crate)` so `decode_residency_tests`' guest knob resolves a name the SAME
/// way this bench does. Two resolvers would be two answers to "which ELF is
/// `ethrex`", which is the question that cost this campaign a day.
pub(crate) fn elf_bytes(name: &str) -> Vec<u8> {
    let root = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("workspace root")
        .join("executor/program_artifacts");
    for dir in ["rust", "asm"] {
        let path = root.join(dir).join(format!("{name}.elf"));
        if let Ok(bytes) = std::fs::read(&path) {
            return bytes;
        }
    }
    panic!("no ELF named {name}");
}

/// The programs to sweep, smallest first: `(elf, private input)`.
///
/// `ethrex` with the 10-transfer block is the repo's reference workload. It runs
/// here **monolithic**, because the multilinear path has no continuations yet,
/// so it is bigger than the epochs the continuation bench proves.
const PROGRAMS: &[(&str, &str)] = &[
    ("sub", ""),
    ("all_instructions_64", ""),
    ("keccak", ""),
    ("ethrex", "ethrex_empty_block"),
    ("ethrex", "ethrex_simple_tx"),
    ("ethrex", "ethrex_bench_4"),
    ("ethrex", "ethrex_10_transfers"),
];

/// A private-input fixture from `executor/tests`, empty for a program that
/// takes none.
///
/// ★ `pub(crate)` for the same reason [`elf_bytes`] is: there must be ONE
/// answer to "which bytes are `ethrex_10_transfers`". The level-0 driver's box
/// test (`lfm::whir_epoch_tests`) loads its fixture through these two rather
/// than re-implementing the path, so the two cannot drift.
pub(crate) fn input_bytes(name: &str) -> Vec<u8> {
    if name.is_empty() {
        return Vec::new();
    }
    let path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("workspace root")
        .join("executor/tests")
        .join(format!("{name}.bin"));
    std::fs::read(&path).unwrap_or_else(|_| panic!("read {}", path.display()))
}

/// What each program costs in trace, without proving anything.
///
/// The multilinear prover's peak is driven by the **stacked** width
/// `num_vars + log2(columns)`: one codeword of `2^(n_stack + log_blowup)` base
/// elements per table, plus the extension-field fold on top of it. Printed here
/// so a program that cannot fit is ruled out before an hour is spent finding
/// out.
#[test]
#[ignore]
fn shapes() {
    println!(
        "\n{:<22} {:>7} {:>8} {:>7} {:>9} {:>12}",
        "program", "tables", "rows", "cols", "n_stack", "cells"
    );
    for &(name, input) in PROGRAMS {
        let label = if input.is_empty() {
            name.to_string()
        } else {
            input.to_string()
        };
        let bytes = elf_bytes(name);
        let inputs = input_bytes(input);
        let elf = match Elf::load(&bytes) {
            Ok(elf) => elf,
            Err(e) => {
                println!("{label:<22} load failed: {e:?}");
                continue;
            }
        };
        let logs = match Executor::new(&elf, inputs.clone()).and_then(Executor::run) {
            Ok(result) => result.logs,
            Err(e) => {
                println!("{label:<22} run failed: {e:?}");
                continue;
            }
        };
        let mut traces = match Traces::from_elf_and_logs(
            &elf,
            &logs,
            &MaxRowsConfig::default(),
            &inputs,
            #[cfg(feature = "disk-spill")]
            stark::storage_mode::StorageMode::Ram,
        ) {
            Ok(traces) => traces,
            Err(e) => {
                println!("{label:<22} trace failed: {e:?}");
                continue;
            }
        };
        let table_counts = traces.table_counts();
        let airs = crate::VmAirs::new(
            &elf,
            &options(),
            false,
            &traces.page_configs,
            &table_counts,
            None,
            true,
            None,
            None,
            None,
        );
        let pairs = airs.air_trace_pairs(&mut traces);

        // The widest stack sets the query count and the biggest single
        // allocation; the **cell total** is what normalises one prover against
        // another running different shapes, which is how the reference system
        // was compared — cells per second.
        let (tables, mut rows, mut cols, mut n_stack) = (pairs.len(), 0usize, 0usize, 0usize);
        let mut cells = 0u64;
        for (_, trace, _) in &pairs {
            let columns = trace.columns_main();
            let height = columns.first().map_or(0, Vec::len);
            let vars = height.trailing_zeros() as usize;
            let stack = vars + columns.len().next_power_of_two().trailing_zeros() as usize;
            cells += (height * columns.len()) as u64;
            if stack > n_stack {
                (rows, cols, n_stack) = (height, columns.len(), stack);
            }
        }
        println!(
            "{label:<22} {tables:>7} {rows:>8} {cols:>7} {n_stack:>9} {:>10.3}e9",
            cells as f64 / 1e9,
        );
    }
}

/// Proves one program and prints wall-clock and proof size.
///
/// `LAMBDA_VM_BENCH_ELF` picks the program, `LAMBDA_VM_BENCH_INPUT` its private
/// input fixture, `LAMBDA_VM_BENCH_BACKEND` which prover runs (`fri`, `whir`,
/// or both). One backend per process is what makes peak RSS attributable, so
/// the memory numbers come from
///
/// ```text
/// LAMBDA_VM_BENCH_BACKEND=whir /usr/bin/time -l cargo test --release ...
/// ```
#[test]
#[ignore]
fn whir_against_fri() {
    let name =
        std::env::var("LAMBDA_VM_BENCH_ELF").unwrap_or_else(|_| "all_instructions_64".into());
    let input = std::env::var("LAMBDA_VM_BENCH_INPUT").unwrap_or_default();
    let backend = std::env::var("LAMBDA_VM_BENCH_BACKEND").unwrap_or_else(|_| "both".into());
    let bytes = elf_bytes(&name);
    let inputs = input_bytes(&input);
    let opts = options();
    let max_rows = MaxRowsConfig::default();
    let threads = std::env::var("RAYON_NUM_THREADS").unwrap_or_else(|_| "all".into());
    let label = if input.is_empty() { &name } else { &input };
    println!("\n{label} — CPU, RAYON_NUM_THREADS={threads}, backend={backend}");

    let mib = |n: usize| n as f64 / (1024.0 * 1024.0);
    let mut fri = None;
    let mut whir = None;

    if backend != "whir" {
        let start = Instant::now();
        let proof = crate::prove_with_options_and_inputs(&bytes, &inputs, &opts, &max_rows)
            .expect("univariate prove");
        let prove = start.elapsed();
        let size = rkyv::to_bytes::<rkyv::rancor::Error>(&proof)
            .expect("serialize")
            .len();
        let start = Instant::now();
        let ok = crate::verify_with_options(&proof, &bytes, &opts, None, None).expect("verify");
        assert!(ok, "the univariate proof must verify");
        fri = Some((prove, start.elapsed(), size));
    }

    if backend != "fri" {
        let start = Instant::now();
        let proof =
            multilinear_prove::prove_with_options_and_inputs(&bytes, &inputs, &opts, &max_rows)
                .expect("multilinear prove");
        let prove = start.elapsed();
        let size = rkyv::to_bytes::<rkyv::rancor::Error>(&proof)
            .expect("serialize")
            .len();
        let start = Instant::now();
        let ok = multilinear_prove::verify_with_options(&proof, &bytes, &opts).expect("verify");
        assert!(ok, "the multilinear proof must verify");
        whir = Some((prove, start.elapsed(), size));
    }

    println!(
        "{:<12} {:>10} {:>10} {:>12}",
        "backend", "prove", "verify", "proof"
    );
    for (tag, run) in [("FRI", &fri), ("WHIR", &whir)] {
        if let Some((prove, verify, size)) = run {
            println!(
                "{tag:<12} {:>9.2}s {:>9.2}s {:>10.2} MiB",
                prove.as_secs_f64(),
                verify.as_secs_f64(),
                mib(*size)
            );
        }
    }
    if let (Some(f), Some(w)) = (fri, whir) {
        println!(
            "{:<12} {:>9.2}x {:>9.2}x {:>10.2}x",
            "WHIR/FRI",
            w.0.as_secs_f64() / f.0.as_secs_f64(),
            w.1.as_secs_f64() / f.1.as_secs_f64(),
            w.2 as f64 / f.2 as f64,
        );
    }
}

/// A whole run proved by epochs: the multilinear continuation against the
/// univariate one, at the same epoch size.
///
/// This is the measurement the multilinear continuation exists for. Wall-clock
/// is the visible number, but the reason is the peak: a monolithic proof holds
/// every table of the run at once and an epoch holds one epoch's. That number
/// comes from the OS, one backend per process:
///
/// ```text
/// LAMBDA_VM_BENCH_BACKEND=whir LAMBDA_VM_BENCH_ELF=ethrex \
///   LAMBDA_VM_BENCH_INPUT=ethrex_10_transfers \
///   /usr/bin/time -v cargo test --release ...
/// ```
///
/// `LAMBDA_VM_BENCH_EPOCH_LOG2` is the epoch length in cycles, the CLI's
/// default (2^20) unless it is set. It is a resource knob, not a property of
/// either prover: both sides get the same one.
/// One transcript-counter line, labelled with the WINDOW it covers.
///
/// ⚠ The window is part of the number. The prover and the verifier each run a
/// transcript, over different work, and only the verifier's is what a recursive
/// verifier replays — so a count quoted without its side cannot be checked
/// against anything.
///
/// ⚠ `states` is NOT part of `squeezes`: a squeeze is a `finalize_reset` that
/// chains its output back in, a state read finalizes a CLONE and advances
/// nothing. There is one state read per grind check, so on the verify line the
/// states column must equal the grind-check count — two independent instruments
/// on one quantity.
#[cfg(feature = "hash-metrics")]
fn print_transcript_counts(window: &str, c: &crypto::hash_metrics::Counts) {
    let (ua, us, ut) = c.transcript_unattributed();
    println!(
        "{:<12} transcript absorbs {}/{} · squeezes {}/{} · states {}/{} (keccak/rpx) · unattributed {}/{}/{}",
        window,
        c.transcript_absorbs_keccak,
        c.transcript_absorbs_rpx,
        c.transcript_squeezes_keccak,
        c.transcript_squeezes_rpx,
        c.transcript_states_keccak,
        c.transcript_states_rpx,
        ua,
        us,
        ut,
    );
    // ★ Its own line, and labelled with the same window, because alignment is a
    // property of the byte stream rather than of a sponge: the same stream under
    // two hashes is misaligned in the same places or in neither, so there is no
    // keccak/rpx pair to print.
    //
    // A real proof reads ZERO here, and that zero is now a property rather than
    // a wish: the counter starts at the statement's padding, which is where the
    // padding's promise starts. The earlier definition counted a statement's own
    // fields too and read 25 on the EQ fixture, which is why the name carries
    // the boundary. `a_statement_that_ends_off_a_boundary_is_still_counted` is
    // what stops the zero from being a counter nobody can bump.
    println!(
        "{:<12} misaligned absorbs after a statement: {} (zero is what a correct \
         pad produces; a non-zero says a statement ended off a field element \
         boundary and everything the verifier re-slices after it straddles two)",
        window, c.transcript_misaligned_absorbs_after_statement,
    );
}

/// ★★ THE TRANSCRIPT PAIR, PINNED — one configuration, both sides.
///
/// # Why a pair and not just the verify line
///
/// The verify line alone is the number recursion cares about, but pinning both
/// makes their DIFFERENCE mutation-checkable, and that difference is a derived
/// quantity rather than a measurement: the verifier runs `owed`, the prover does
/// not, and `owed` is `Sum roots.len()` absorbs and `2 x epochs` squeezes. So a
/// change that moves one side without the other fails loudly here instead of
/// silently re-opening a question that took two lanes and a wrong candidate to
/// close.
///
/// The 2 squeezes per `owed` call are not "two samples, two squeezes": a cubic
/// element is three 8-byte draws from a 32-byte buffer, so the first sample
/// squeezes once and leaves a group over, and the second spends the leftover and
/// squeezes again. Three not dividing four is the whole reason it is two.
///
/// # WARNING: the guard is the ELF's sha256, not its name
///
/// Two builds of the same guest, from the same sources, in two worktrees, share
/// a name and differ in bytes. That is not hypothetical: a 0.23% difference in
/// these very counts was read as a model error before it was traced to a
/// different build of "the same" guest — the arms' ELF touches genesis pages
/// (`0x280000`, `0x680000`) that another build does not, which moves the
/// GLOBAL_MEMORY set and with it the chain shapes these counts are made of.
///
/// So a rebuild must not fail this assert — it must SKIP it, out loud, naming
/// the sha it saw. A silent pass and an absent assert are the same thing.
///
/// # Configuration is part of the constant
///
/// The counts are a function of the ELF, the input and the epoch size, so all
/// three are in the guard. Measured on FAST under `cuda,hash-metrics`, and
/// independently reproduced by lane V1's shape-derived closed form, which
/// predicts the verify triple exactly from the table shapes — two derivations,
/// one number.
///
/// # ...and so is the BRANCH, which is why the absorbs are not one number
///
/// The absorb columns carry a term that differs between lineages — one `u64`
/// per per-table count, per epoch statement — so they are pinned as a base plus
/// `EPOCHS * NUM_TABLE_KINDS`, with the kind count read from the struct. The
/// bases were taken on the seam branch (`whir/rpx`, the `c73568f4` measurement
/// plus this branch's computed statement padding); the merged lineage, which
/// binds one count more, gets its own totals from the same bases without
/// editing anything here. See [`transcript_pin::PROVE_BASE_ABSORBS`].
#[cfg(feature = "hash-metrics")]
mod transcript_pin {
    /// sha256 of the guest ELF these counts were measured against.
    ///
    /// MEASURED, and it has to say where: `sha256sum` on the box fixture
    /// `ethrex_8f826601.elf`. The previous value agreed with this one for
    /// exactly 16 hex characters and was invented for the other 48 — every
    /// message that carried the sha carried a 16-char prefix, and the tail was
    /// written to look like a measurement. The guard then skipped on the pinned
    /// guest itself, and the skip line printed `[..16]` of both sides, which is
    /// precisely the width at which a fabricated tail still agrees.
    ///
    /// A guard on a value nobody measured to full width is a guard on a guess.
    /// Anything shortened for a message is a display; the constant is the
    /// measurement.
    pub const ELF_SHA256: &str = "8f826601776d4085cbb6fbf0302fe8d8d5d1be7940ac1aaca24899c6244ec80a";
    pub const ELF_LEN: usize = 3_948_504;
    pub const EPOCH_LOG2: u32 = 21;

    /// ★★★ THE UNIFORM TABLE CAP these counts were measured at, and it is part
    /// of the pin's IDENTITY rather than a footnote.
    ///
    /// ⛔ THE DEFECT THIS EXISTS AGAINST, measured: run lb11 proved the pinned
    /// guest at the pinned epoch size and failed with `the PROVE-side
    /// transcript counts moved — left: (407292, 149121, 3411) right: (585245,
    /// 186256, 3251)`. Nothing was wrong with either side. The constants had
    /// been measured with `LAMBDA_VM_MAX_ROWS_LOG2` UNSET — the production
    /// per-table caps, which chunk CPU into four, MEMW_R into five and LT into
    /// two, so an epoch carries 34 tables — and lb11 ran at the record posture,
    /// a uniform `2^21`, where the same block's epoch carries **27**. Same
    /// guest, same epoch size, different epochs; and [`super::pin_applies`]
    /// could not tell, so a run at a posture nobody pinned was "the pinned
    /// configuration" by the pin's own identity.
    ///
    /// The states column is where it is legible without any of this context:
    /// 3,251 is the cap-less grind count and 3,411 the record posture's, and
    /// the grind count is one state read per check.
    ///
    /// ⇒ The RECORD posture is the pinned one. A run at any other cap — the
    /// production values included — SKIPS and says so, naming both caps. It
    /// never passes and it never panics: another posture is not a defect, it is
    /// a different measurement.
    pub const MAX_ROWS_LOG2: u32 = 21;

    /// Epoch proofs in the pinned run. Part of the measured shape, like the ELF
    /// and the epoch size: 2^21 epochs over this guest is fifteen of them, which
    /// is also where `OWED`'s thirty squeezes come from (two per epoch call).
    pub const EPOCHS: u64 = 15;

    /// ★★ THE ABSORB COUNTS ARE A BASE PLUS A PER-BRANCH TERM, and the term is
    /// read from the struct rather than written down.
    ///
    /// Every epoch statement binds one `u64` per per-table count
    /// (`statement::absorb_table_counts`), so the absorb total carries
    /// `EPOCHS * NUM_TABLE_KINDS`. That count is **not a constant of the
    /// protocol**: the per-table campaign adds `TableCounts::blake3`, so the
    /// seam lineage absorbs fourteen per epoch and this merged one fifteen. The
    /// box
    /// discovered that as a pin FAILURE (+15 on both sides at the merged base),
    /// and the term was legitimate — the BLAKE3 table is conditional and its
    /// count is the one entry a verifier cannot derive, which is why per-table
    /// bumped both domain tags for it.
    ///
    /// A pin carrying `583_940` would therefore be a constant describing one
    /// branch while claiming to describe the protocol. The bases below are
    /// branch-independent; the branch supplies its own kind count.
    ///
    /// ⛔ AND THE BASE IS POSTURE-SPECIFIC, which the paragraph above does NOT
    /// cover. "Branch-independent" is true across TABLE-KIND branches and false
    /// across table-CHUNKING postures: the base counts the per-table walk and
    /// the chains of every epoch, and how many of each an epoch has is what
    /// [`MAX_ROWS_LOG2`] decides. The cap-less pair was `583_730 / 583_867`;
    /// the record posture's is below, and the two are 177,953 apart on the
    /// prove line. Neither is wrong — they are two postures, and the identity
    /// now carries which.
    ///
    /// MEASURED: run lb11 at `whir/lfm-l0-main-sync` @ `892c7d1bc` on FAST
    /// (`--features cuda,hash-metrics`, `LAMBDA_VM_WHIR_HASH=rpx`, the budget
    /// to the driver's query, never-purge, guest `8f826601…ec80a` 3,948,504 B,
    /// input `573004e6…f17f` 1,110,183 B, `LAMBDA_VM_MAX_ROWS_LOG2=21`), which
    /// read PROVE `(407292, 149121, 3411)` and VERIFY `(407452, 149151, 3411)`.
    /// The base is that total less the terms this module derives —
    /// [`table_count_absorbs`] at 15 x 21 = 315, the derived root's 15 (prove)
    /// or 30 (verify), and `15 x 79` for the prepared opening — so a branch
    /// that changes a derived term still moves the totals without anyone
    /// editing a literal. It carries NO genesis-stack term: lb11's guest had
    /// none, and the stack's own cost is [`genesis_stack_terms`].
    ///
    /// ★★ AND THAT IS WHY CARRYING THESE ONTO THIS BRANCH IS A VERIFIED MOVE
    /// RATHER THAN A NEW LITERAL. Composed with the terms this module derives
    /// AND with the genesis stack's, they reproduce runs lb17/lb18 — the
    /// measurement taken at this lineage's own tip — on all six numbers and
    /// with no residue:
    ///
    /// ```text
    /// prove   absorbs  405_777 + 315 + 15 + 15x79 + 77 = 407_369   measured 407_369
    /// prove   squeezes 146_091 + 15x202 + 200          = 149_321   measured 149_321
    /// states  3_156 + 15x17 + 17                       =   3_428   measured   3_428
    /// verify  absorbs  405_922 + 315 + 30 + 15x79 + 77 = 407_529   measured 407_529
    /// verify  squeezes 146_121 + 15x202 + 200          = 149_351   measured 149_351
    /// ```
    ///
    /// A carried constant that did not compose would be a number answering a
    /// retired question; these are the only values of the base for which the
    /// derived terms land on the measurement.
    ///
    /// ⛔ WHAT IS NOT DERIVED, said plainly: the base itself. Deriving it needs
    /// the whole per-table walk's transcript cost at this posture, which no
    /// form in this tree computes.
    pub const PROVE_BASE_ABSORBS: u64 = 405_777;
    /// The verify side's base. See [`PROVE_BASE_ABSORBS`].
    pub const VERIFY_BASE_ABSORBS: u64 = 405_922;

    /// Absorbs the per-table counts contribute to a whole continuation proof.
    pub const fn table_count_absorbs() -> u64 {
        EPOCHS * crate::statement::NUM_TABLE_KINDS as u64
    }

    /// The DECODE group's shape at the pinned guest: columns, and log2 of the
    /// rows its instruction table fills.
    ///
    /// ⚠ NOT A CONSTANT OF THE PROTOCOL — it is this guest's instruction table,
    /// so [`super::check_transcript_pins`] DERIVES it from the ELF and asserts
    /// it before comparing any count. A guest whose program grew past 2^20
    /// instructions then fails by name, instead of moving every number below by
    /// an amount that would read as a protocol change.
    pub const DECODE_PREPARED_SHAPE: (usize, usize) = (5, 20);

    /// Eight-byte candidates one 32-byte sponge squeeze hands out, which is why
    /// a round's `num_queries` draws cost `ceil(Q / 4)` squeezes and not `Q`.
    pub const CANDIDATES_PER_SQUEEZE: u64 = 4;

    /// ★★ WHAT ONE PREPARED OPENING COSTS THE TRANSCRIPT, PER EPOCH — derived
    /// from the round structure, not from the box.
    ///
    /// The landing gate before run a2v priced this at "+1 absorb per epoch": the
    /// derived root's absorb, and nothing else. But opening a committed
    /// polynomial in WHIR *is a chain*, and W1-B opens DECODE's out-of-band
    /// commitment at DECODE's reduced point once per epoch. The box read
    /// +79 absorbs / +202 squeezes / +17 states an epoch on both hashes. What
    /// follows is that number reached from the code instead, so the next shape
    /// change moves it on its own.
    ///
    /// `stacked_eval::verify` absorbs one field element per
    /// `layout.placements()` entry — the group's columns — then draws ONE
    /// batching challenge, then runs `layout.num_polys()` chains at
    /// `layout.n_stack()`. For this group the stacker puts all five columns in
    /// one polynomial, so it is one chain and five absorbs.
    ///
    /// Each chain is `whir_chain::verify_weighted` over
    /// `config.schedule(n_stack)`. Per round, read off that loop:
    ///
    /// * `check_grind(folding)` — one state read, one nonce absorbed;
    /// * `sumcheck::verify_rounds(.., degree 2, ..)` — two evaluations absorbed
    ///   and one challenge drawn per variable, so `2K` absorbs and `K` squeezes
    ///   over the chain, `K` being the schedule's sum;
    /// * a non-final round also absorbs the successor root and the
    ///   out-of-domain value, draws `z0` and `gamma`, and grinds twice more:
    ///   four absorbs, two squeezes, two states, `R - 1` times;
    /// * the final round absorbs the final value and grinds once: two absorbs,
    ///   one state, and no `z0` or `gamma` because there is no successor;
    /// * every round draws `num_queries` positions with `sample_u64` and
    ///   absorbs nothing (`whir_round::verify`, `verify_final`).
    ///
    /// ⚠ THE SQUEEZE COLUMN CARRIES AN ASSUMPTION THE ABSORB COLUMN DOES NOT:
    /// that each field-element sample costs one squeeze and that consecutive
    /// `u64` draws pack [`CANDIDATES_PER_SQUEEZE`] to a squeeze. That is what
    /// makes the query term `R * ceil(Q / 4)` rather than `R * Q`. It is the
    /// reading the box confirmed at this shape; a buffer that straddled
    /// differently would show up in this column and nowhere else.
    pub fn prepared_opening_per_epoch(shape: (usize, usize)) -> (u64, u64, u64) {
        let (layout, config) = prepared_group(shape);
        let schedule = config.schedule(layout.n_stack());

        let rounds = schedule.len() as u64;
        let folded: u64 = schedule.iter().sum::<usize>() as u64;
        let queries = config.num_queries as u64;
        let per_round_query_squeezes = queries.div_ceil(CANDIDATES_PER_SQUEEZE);

        let grinds = 3 * rounds - 1;
        let chain_absorbs = grinds + 2 * folded + 2 * (rounds - 1) + 1;
        let chain_squeezes = folded + 2 * (rounds - 1) + rounds * per_round_query_squeezes;

        let polys = layout.num_polys() as u64;
        (
            layout.placements().len() as u64 + polys * chain_absorbs,
            1 + polys * chain_squeezes,
            polys * grinds,
        )
    }

    /// The layout and chain config the DECODE group is committed and opened
    /// under — the same two values `decode_prepared_for` builds, from the same
    /// functions, so nothing here can describe a group the prover does not use.
    pub fn prepared_group(
        shape: (usize, usize),
    ) -> (
        multilinear::stacking::StackedLayout,
        multilinear::whir_chain::ChainConfig,
    ) {
        let (columns, num_vars) = shape;
        (
            stark::multilinear_table::global_layout(&[(columns, num_vars)])
                .expect("the DECODE group's layout"),
            crate::multilinear_continuation::decode_prepared_config(columns, num_vars),
        )
    }

    /// The DENSE-GENESIS STACK's shape, as the run's own plan states it.
    ///
    /// ⚠ NOT A THRESHOLD, AND THIS TYPE CARRIES NO OPINION ABOUT ONE. Which
    /// pages are dense enough to be stacked is decided in exactly one place,
    /// `continuation::genesis_stack_plan`, and it is READ from there through
    /// the verifier's own `global_airs_for(..).genesis_stack()`. A second
    /// spelling of the rule here would be a pin describing a stack the prover
    /// does not build — which is the whole failure mode this module exists
    /// against, one level up.
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub struct StackShape {
        /// Stacked columns: `GenesisStackPlan::at`'s length, which is
        /// `PAGE_PREPROCESSED_COLUMNS` — `OFFSET` and `INIT` — per dense page.
        pub columns: usize,
        /// Each column's height in variables. Every genesis page is one page
        /// tall, so the stack is a rectangle and this is `PAGE_NUM_VARS`.
        pub num_vars: usize,
    }

    /// ★★ WHAT THE GENESIS STACK COSTS THE WHOLE RUN'S TRANSCRIPT — `(0, 0, 0)`
    /// when the run has no dense page, derived from the stack's SHAPE otherwise.
    ///
    /// ⛔ THE CONTROL THIS ANSWERS, measured: runs lb17 and lb18 read
    /// `+77 absorbs / +200 squeezes / +17 states on BOTH sides` and `+1 device
    /// commit` against the pinned pair, at two different tips with identical
    /// numbers. The pin's constants were taken before W1i put the stacked INIT
    /// polynomial of the dense genesis pages into the cross-epoch statement;
    /// what follows is those four numbers reached from the code, so the next
    /// shape change moves them on its own.
    ///
    /// ★ ONCE FOR THE RUN, NOT ONCE PER EPOCH. The stack belongs to the
    /// CROSS-EPOCH proof — `prove_global` commits it and hands it to
    /// `multi_prove` as that proof's prepared opening — so unlike
    /// [`prepared_opening_per_epoch`] this term is not multiplied by
    /// [`EPOCHS`].
    ///
    /// ★ AND THE SAME ON BOTH SIDES, which is why it is not part of [`OWED`].
    /// DECODE's derived root costs the verifier twice because the EPOCH path
    /// replays the roots block in `multilinear_continuation::owed`; the
    /// cross-epoch path has no replay, so the stack's root is absorbed exactly
    /// once on each side and the whole opening is run by both. A term that
    /// moved one line and not the other would show up in
    /// `the_pinned_constants_differ_by_owed` and nowhere else.
    ///
    /// The terms, each against the site that pays it:
    ///
    /// * the stack's roots in the cross-epoch roots block — one per stacked
    ///   polynomial, absorbed by `multilinear_table::absorb_roots` from the
    ///   `prepared_roots` list `multi_prove` builds;
    /// * one claimed value per stacked column, absorbed by
    ///   `stacked_eval::prove`, then ONE batching challenge drawn — the same
    ///   two terms [`prepared_opening_per_epoch`] opens with;
    /// * then `layout.num_polys()` chains through `whir_chain`, at
    ///   `config.schedule(layout.n_stack())`, priced exactly as the prepared
    ///   opening's chains are and for the same reasons — see that function's
    ///   round-by-round reading, which is the one this shares.
    ///
    /// ⚠ The roots block's own three challenges (`z`, `alpha`, `beta`) are
    /// drawn ONCE per proof however many roots were absorbed, so the stack adds
    /// to the absorb column there and nothing to the squeeze column. A term
    /// added to both would be an extra squeeze this pin would then be unable to
    /// see.
    pub fn genesis_stack_terms(
        shape: Option<StackShape>,
        config: &multilinear::whir_chain::ChainConfig,
    ) -> (u64, u64, u64) {
        let Some(shape) = shape else {
            return (0, 0, 0);
        };
        let layout = genesis_stack_layout(shape);
        let schedule = config.schedule(layout.n_stack());

        let rounds = schedule.len() as u64;
        let folded: u64 = schedule.iter().sum::<usize>() as u64;
        let queries = config.num_queries as u64;
        let per_round_query_squeezes = queries.div_ceil(CANDIDATES_PER_SQUEEZE);

        let grinds = 3 * rounds - 1;
        let chain_absorbs = grinds + 2 * folded + 2 * (rounds - 1) + 1;
        let chain_squeezes = folded + 2 * (rounds - 1) + rounds * per_round_query_squeezes;

        let polys = layout.num_polys() as u64;
        (
            polys + layout.placements().len() as u64 + polys * chain_absorbs,
            1 + polys * chain_squeezes,
            polys * grinds,
        )
    }

    /// The layout the genesis stack is committed under — from the same
    /// `global_layout` call `genesis_prepared_for` makes, over the same
    /// rectangle, so this cannot describe a stack the prover does not build.
    pub fn genesis_stack_layout(shape: StackShape) -> multilinear::stacking::StackedLayout {
        stark::multilinear_table::global_layout(&[(shape.columns, shape.num_vars)])
            .expect("the genesis stack's layout")
    }

    /// Device commits the genesis stack costs the whole run: its held
    /// commitment, plus the chain's successor codewords.
    ///
    /// ⚠ ONE TERM MORE THAN [`prepared_fold_commits_per_epoch`] BECAUSE THE
    /// HELD HALF IS NOT SHARED. DECODE's commitment is a function of the ELF
    /// and is built once and reused by all fifteen epochs, so `commits`
    /// counts it once for the run. The stack's is built once per `prove_global`
    /// CALL and belongs to that one proof — and counting held commitments with
    /// a `find_map` over the proof list, which stops at the first proof that
    /// has one, is exactly how the stack's went uncounted and the model read
    /// 1,187 against a counter of 1,188.
    pub fn genesis_stack_commits(
        shape: Option<StackShape>,
        config: &multilinear::whir_chain::ChainConfig,
    ) -> u64 {
        let Some(shape) = shape else {
            return 0;
        };
        let layout = genesis_stack_layout(shape);
        let rounds = config.schedule(layout.n_stack()).len() as u64;
        layout.num_polys() as u64 * rounds
    }

    /// Device commits one prepared opening costs per epoch: the chain's
    /// successor codewords, `R - 1` of them per stacked polynomial.
    pub fn prepared_fold_commits_per_epoch(shape: (usize, usize)) -> u64 {
        let (layout, config) = prepared_group(shape);
        let rounds = config.schedule(layout.n_stack()).len() as u64;
        layout.num_polys() as u64 * (rounds - 1)
    }

    /// The commitment itself: one commit per stacked polynomial, built once for
    /// the whole run and held across the epochs — which is the count
    /// `decode_residency_tests` reads off the production path.
    pub fn prepared_commitments(shape: (usize, usize)) -> u64 {
        prepared_group(shape).0.num_polys() as u64
    }

    /// Roots DECODE's out-of-band commitment contributes to one epoch's roots
    /// block — one, and it is a property of the derivation rather than of this
    /// guest.
    ///
    /// `decode_prepared_from_columns` commits the five columns under
    /// `global_layout(&[(5, rows)])`, whose `n_stack` is `ceil_log2(5 * rows)`,
    /// so all five always fit one stacked polynomial and the commitment always
    /// has exactly one root. `decode_prepared_tests` pins that shape; this
    /// constant is what makes the pin move if it ever stops holding.
    pub const DERIVED_ROOTS_PER_EPOCH: u64 = 1;

    /// Absorbs DECODE's derived root contributes to the PROVE line: one per
    /// epoch, inside `multi_prove`'s roots block.
    pub const fn prove_derived_root_absorbs() -> u64 {
        EPOCHS * DERIVED_ROOTS_PER_EPOCH
    }

    /// ...and to the VERIFY line, which is TWICE that and not the same number.
    ///
    /// The verifier absorbs the derived root in `multi_verify`'s roots block
    /// AND again in the `owed` replay, which has to see the same block or it
    /// draws challenges no table is checked at. So the term is doubled here and
    /// the second half of it is also what [`OWED`] grows by.
    ///
    /// ⚠ Writing one term for both sides is the mistake this split exists to
    /// prevent, and it was made: a single `derived_root_absorbs()` on both lines
    /// put VERIFY at `584_092`, and `the_pinned_constants_differ_by_owed` caught
    /// it because the pair then differed by 137 while `OWED` said 152.
    pub const fn verify_derived_root_absorbs() -> u64 {
        2 * EPOCHS * DERIVED_ROOTS_PER_EPOCH
    }

    /// The squeezes and states a continuation cost before the prepared opening
    /// and the genesis stack, at the RECORD posture — run lb11's measurement,
    /// the same one [`PROVE_BASE_ABSORBS`] comes from and at the same cap.
    ///
    /// The two sides differ by `owed`'s thirty squeezes and by nothing else, and
    /// neither side reads a transcript state outside a grind check, which is why
    /// the state base is one number for both.
    ///
    /// ⚠ POSTURE-SPECIFIC, like the absorb base. The cap-less pair was
    /// `183_226 / 183_256` with `2_996` states; a run at the production caps
    /// SKIPS rather than comparing against these.
    pub const PROVE_BASE_SQUEEZES: u64 = 146_091;
    /// See [`PROVE_BASE_SQUEEZES`].
    pub const VERIFY_BASE_SQUEEZES: u64 = 146_121;
    /// See [`PROVE_BASE_SQUEEZES`]. One state read per grind check, both sides.
    pub const BASE_STATES: u64 = 3_156;

    /// The genesis stack as a run states it: its shape and the chain config the
    /// cross-epoch proof argues at, or `None` for a run with no dense page.
    ///
    /// ⚠ CARRIED AS ONE VALUE because the two halves have to come from the same
    /// derivation. The shape decides `n_stack` and the config decides the
    /// schedule that `n_stack` is folded by; taking them from two places is how
    /// a count comes to describe a stack at one posture folded at another.
    #[derive(Clone, Debug)]
    pub struct Stack {
        pub shape: Option<StackShape>,
        pub config: multilinear::whir_chain::ChainConfig,
    }

    impl Stack {
        pub fn terms(&self) -> (u64, u64, u64) {
            genesis_stack_terms(self.shape, &self.config)
        }
    }

    /// (absorbs, squeezes, states) after `prove_continuation`.
    ///
    /// ⚠ A FUNCTION, not a constant, because the opening's terms come out of
    /// `schedule()` and that allocates. The shape it is evaluated at is asserted
    /// against the ELF before any comparison is made.
    ///
    /// ⚠ AND `stack` IS THE RUN'S, not a constant of this guest. The bases were
    /// measured by a run whose genesis was entirely sparse, so a `None` here
    /// reproduces them exactly; the block's three dense pages add
    /// [`genesis_stack_terms`] on top. Hard-coding the block's stack would make
    /// this pin unable to describe the very run its bases came from.
    pub fn prove(shape: (usize, usize), stack: &Stack) -> (u64, u64, u64) {
        let (a, s, t) = prepared_opening_per_epoch(shape);
        let (ga, gs, gt) = stack.terms();
        (
            PROVE_BASE_ABSORBS
                + table_count_absorbs()
                + prove_derived_root_absorbs()
                + EPOCHS * a
                + ga,
            PROVE_BASE_SQUEEZES + EPOCHS * s + gs,
            BASE_STATES + EPOCHS * t + gt,
        )
    }

    /// ...and after `verify_continuation`. The difference is `owed`, nothing else.
    ///
    /// ★ The opening's term is the SAME on both sides and is NOT part of `owed`:
    /// the prover runs the opening too, and `owed` replays only the roots block.
    /// So the new term moves both lines by the same amount and
    /// `the_pinned_constants_differ_by_owed` stays a live check rather than one
    /// the new term could have absorbed.
    ///
    /// ★ The genesis stack's term is the same on both sides for a SECOND
    /// reason, and the two must not be conflated: `owed` is the EPOCH path's
    /// replay, and the stack lives in the cross-epoch proof, which has no
    /// replay at all.
    pub fn verify(shape: (usize, usize), stack: &Stack) -> (u64, u64, u64) {
        let (a, s, t) = prepared_opening_per_epoch(shape);
        let (ga, gs, gt) = stack.terms();
        (
            VERIFY_BASE_ABSORBS
                + table_count_absorbs()
                + verify_derived_root_absorbs()
                + EPOCHS * a
                + ga,
            VERIFY_BASE_SQUEEZES + EPOCHS * s + gs,
            BASE_STATES + EPOCHS * t + gt,
        )
    }

    /// Device commits a whole continuation makes, READ OFF THE BUNDLE.
    ///
    /// ★ NO MEASURED BASE. An earlier draft carried the box's 1,050 as a
    /// literal; it is not a constant, it is a sum the proof states. A group's
    /// opening is a `StackedProof`, each of its polynomials a `ChainProof`, and
    /// `rounds.len()` is one initial commitment plus `R - 1` successor
    /// codewords — one commit per round. Summed over every epoch's groups and
    /// the cross-epoch proof's, that IS 1,050 on the pinned fixture, which is
    /// what `whir_epoch_shapes` prints as "chains 154 rounds 1050".
    ///
    /// DECODE's prepared opening is the one term that is not per-chain: its
    /// polynomial is committed ONCE for the run and held, so it contributes
    /// `rounds.len() - 1` folds per epoch plus a single commitment — the
    /// residency claim, arriving here as arithmetic.
    ///
    /// ⛔ AND THE GENESIS STACK IS A SECOND HELD COMMITMENT WITH A DIFFERENT
    /// SCOPE, which is the defect this signature exists against. Both openings
    /// arrive in a `MultiProof::preprocessed`, so a model that walked a flat
    /// list of proofs counted the stack's FOLDS — they are per opening, and
    /// there is one opening — while counting its HELD commitment not at all:
    /// the held term was a `find_map`, which stops at the first proof carrying
    /// a `preprocessed`, and the epoch proofs come first. Runs lb17 and lb18
    /// read that as `left: 1188 / right: 1187`, one commit apart, at two tips
    /// with identical counters. The two scopes are now two arguments and two
    /// terms: DECODE's commitment is a function of the ELF and is held across
    /// every epoch; the stack's is built once per `prove_global` call and
    /// belongs to that one proof. Neither can be inferred from slice order.
    ///
    /// ⚠ AND THE ASSUMPTION THIS CARRIES, because it is shape-dependent and
    /// silent. The counter counts DEVICE commits, and a polynomial too small for
    /// the device is committed on the host, where a FOLD commit is counted
    /// nowhere at all. `decode_residency_tests` measured exactly that: at a
    /// 5 x 16 DECODE group the model reads 11 and the device counter reads 9.
    /// This model therefore assumes every chain reached the device, which holds
    /// here only because the pinned guest's polynomials are far above that
    /// boundary — and the ELF sha guard above is what makes "the pinned guest"
    /// something this function may assume.
    ///
    /// ⚠ `cuda`-only, because it models a counter that exists nowhere else. The
    /// SHAPE half of the arithmetic is still laptop-checked —
    /// `the_prepared_opening_is_the_schedule_the_shape_implies` pins the
    /// prepared group's folds per epoch and its single held commitment — and
    /// what runs only on the box is the summation over the bundle's chains.
    #[cfg(feature = "cuda")]
    pub type Proof =
        stark::multilinear_table::MultiProof<crate::test_utils::F, crate::test_utils::E>;

    /// One commit per chain round, over a proof's own commitment groups.
    #[cfg(feature = "cuda")]
    fn chain_commits(proof: &Proof) -> u64 {
        proof
            .columns
            .iter()
            .flat_map(|group| &group.polys)
            .map(|poly| poly.rounds.len() as u64)
            .sum()
    }

    /// The successor codewords one prepared opening commits: `R - 1` per
    /// stacked polynomial. The initial commitment is NOT here — it is held, and
    /// who holds it is the caller's to say.
    #[cfg(feature = "cuda")]
    fn prepared_fold_commits(proof: &Proof) -> u64 {
        proof
            .preprocessed
            .iter()
            .flat_map(|opening| &opening.polys)
            .map(|poly| poly.rounds.len() as u64 - 1)
            .sum()
    }

    /// The held commitment a proof's prepared opening was made against: one per
    /// stacked polynomial.
    #[cfg(feature = "cuda")]
    fn held_commits(proof: &Proof) -> u64 {
        proof
            .preprocessed
            .as_ref()
            .map(|opening| opening.polys.len() as u64)
            .unwrap_or(0)
    }

    #[cfg(feature = "cuda")]
    pub fn commits(epochs: &[&Proof], global: &Proof) -> u64 {
        let chains: u64 =
            epochs.iter().copied().map(chain_commits).sum::<u64>() + chain_commits(global);
        let folds: u64 = epochs
            .iter()
            .copied()
            .map(prepared_fold_commits)
            .sum::<u64>()
            + prepared_fold_commits(global);
        // DECODE's commitment is a function of the ELF: built once and held,
        // however many epochs open it. Taken from the first epoch that carries
        // one, because that is the one commitment all of them share.
        let decode_held: u64 = epochs
            .iter()
            .copied()
            .find(|p| p.preprocessed.is_some())
            .map(held_commits)
            .unwrap_or(0);
        // The genesis stack's commitment is built once per `prove_global` call
        // and is this proof's alone, so it is counted here and not folded into
        // the term above.
        let stack_held = held_commits(global);
        chains + folds + decode_held + stack_held
    }

    /// `owed`'s absorbs BEFORE W1-B's out-of-band opening existed: `Sum
    /// roots.len()` over the 15 epoch calls.
    ///
    /// ⚠ POSTURE-SPECIFIC, like the bases. An epoch's carried roots are one per
    /// stacked polynomial of every commitment group — which is exactly its
    /// CHAIN count — so this is `Sum chains(epoch)` over the block's fifteen
    /// epochs and nothing else, and how many chains an epoch has is what
    /// [`MAX_ROWS_LOG2`] decides. At the record posture the fifteen epochs give
    /// 9, 9, 9, 12, 12, 11, 10, 9, 9, 9, 9, 9, 9, 10, 9, which sums to **145**;
    /// at the cap-less posture it was 137.
    ///
    /// ★ AND IT IS CHECKED AGAINST THE RUN'S OWN PROOF rather than left as a
    /// literal: [`super::check_transcript_pins`] sums `roots.len()` over the
    /// bundle's fifteen epoch proofs and asserts it equals this. That is the
    /// same quantity from the object the measurement came out of, so a posture
    /// change reddens here by name instead of arriving as "the counts moved".
    ///
    /// ⚠ The genesis stack adds NOTHING to this. `owed` is the epoch path's
    /// replay and the stack lives in the cross-epoch proof, which has none.
    pub const OWED_CARRIED_ABSORBS: u64 = 145;

    /// `owed`'s own cost, stated rather than left as a subtraction: the carried
    /// roots plus DECODE's derived one, `2 x 15` squeezes, and no state read.
    ///
    /// ⚠ The squeeze count is TWO per call and not three. `owed` shares the
    /// roots block's absorb half and spells its own draws, because a fork that
    /// is discarded must not pay a squeeze nobody reads — and `hash_metrics`
    /// counts squeezes on a clone like any other transcript, so a third draw
    /// would show up right here as `3 x 15`.
    pub const OWED: (u64, u64, u64) = (
        OWED_CARRIED_ABSORBS + prove_derived_root_absorbs(),
        2 * EPOCHS,
        0,
    );
}

/// Whether the pinned counts describe THIS run.
///
/// Pure, and taking the sha as a string so the refusal can be tested without
/// forging an ELF: a sha that agrees on a prefix and differs in the tail is one
/// `format!` away, which is the case that actually occurred.
#[cfg(feature = "hash-metrics")]
fn pin_applies(sha: &str, len: usize, epoch_size_log2: u32, max_rows_log2: Option<u32>) -> bool {
    sha == transcript_pin::ELF_SHA256
        && len == transcript_pin::ELF_LEN
        && epoch_size_log2 == transcript_pin::EPOCH_LOG2
        && max_rows_log2 == Some(transcript_pin::MAX_ROWS_LOG2)
}

/// How the pinned cap and a run's cap are SHOWN — a cap-less run has to be
/// legible as such.
///
/// `None` prints as the production caps rather than as an empty field, because
/// "unset" is not the absence of a posture: it selects the per-table values,
/// which is the posture the old constants were measured at and the one that
/// made this pin fail while looking like it applied.
#[cfg(feature = "hash-metrics")]
fn cap_label(max_rows_log2: Option<u32>) -> String {
    match max_rows_log2 {
        Some(n) => format!("uniform 2^{n}"),
        None => "UNSET (the production per-table caps)".to_string(),
    }
}

/// The line a skipped pin prints.
///
/// FULL 64 hex on BOTH sides, never a prefix. The truncated version of this
/// line is why a wrong constant survived a box run: it showed `[..16]` of each,
/// the two agreed there, and the mismatch it existed to report was invisible in
/// its own output. A diagnostic that can agree while the values differ is not a
/// diagnostic.
#[cfg(feature = "hash-metrics")]
fn pin_skip_line(
    sha: &str,
    len: usize,
    epoch_size_log2: u32,
    max_rows_log2: Option<u32>,
) -> String {
    format!(
        "{:<12} transcript pin SKIPPED - elf sha {} ({} bytes, epoch 2^{}, table cap {}); \
         pinned {} ({} bytes, epoch 2^{}, table cap {})",
        "WHIR",
        sha,
        len,
        epoch_size_log2,
        cap_label(max_rows_log2),
        transcript_pin::ELF_SHA256,
        transcript_pin::ELF_LEN,
        transcript_pin::EPOCH_LOG2,
        cap_label(Some(transcript_pin::MAX_ROWS_LOG2)),
    )
}

/// What a pinned RUN states about itself, beyond the ELF — the facts the
/// derived terms are evaluated at.
///
/// ⚠ AN `Option` AT THE CALL, AND THE `None` IS NOT A DEFAULT. The guard half
/// of the pin has to be reachable without a proof (see
/// [`the_transcript_pin_skips_a_guest_it_does_not_recognise`]), and the
/// assertion half cannot run without one. So a `None` that gets PAST the guard
/// panics rather than substituting zeros: a stack term silently taken as zero
/// on a run that has a stack is the exact shape of a pin that passes while
/// describing something else.
///
/// ⚠ OWNED, NOT BORROWED FROM THE BUNDLE, and the reason is where the pins are
/// called from: after the results table, so a red pin costs nothing that was
/// measured — by which point the bundle has been dropped. Every field is read
/// off the proof inside the measured arm and carried out as a number.
#[cfg(feature = "hash-metrics")]
struct RunFacts {
    /// Epoch proofs the run produced. Three pinned terms are multiplied by it.
    epochs: u64,
    /// `Sum roots.len()` over those epoch proofs — one root per chain, which is
    /// `owed`'s carried half.
    carried_roots: u64,
    /// The cross-epoch proof's prepared opening, as the PROOF states it: one
    /// entry per stacked polynomial. Empty when the proof carries none.
    stack_chains: Vec<StackChain>,
    stack: transcript_pin::Stack,
}

/// One chain of the cross-epoch prepared opening, as the proof states it.
#[cfg(feature = "hash-metrics")]
struct StackChain {
    /// Variables folded in each round — `sumcheck.len()` per round, which is
    /// the schedule the chain actually ran.
    folds: Vec<usize>,
    /// Positions opened per round, from the first round's block.
    queries: usize,
}

/// The cross-epoch proof's chain config and genesis-stack plan, DERIVED THE WAY
/// THE VERIFIER DERIVES THEM.
///
/// ⛔ THROUGH `global_airs_for`, AND THAT IS THE POINT. The threshold that
/// decides which genesis pages are stacked lives in `genesis_stack_plan`, and
/// the AIR set is the one object that knows the bookend offset its
/// `PreparedColumn` indices are relative to. Reaching either of them another
/// way here — re-calling `global_memory_configs`, or re-evaluating the density
/// rule — would be a second derivation of the very thing the pin claims to
/// describe, which is what that function's own doc refuses. The page configs
/// come from the ELF and the public shape comes from the bundle, exactly as
/// `verify_global_bookends` takes them.
#[cfg(feature = "hash-metrics")]
fn global_stack(
    elf_bytes: &[u8],
    bundle: &crate::multilinear_continuation::ContinuationProof,
    opts: &crate::ProofOptions,
) -> transcript_pin::Stack {
    let elf = Elf::load(elf_bytes).expect("the proven guest loads");
    let air_set = crate::multilinear_continuation::global_airs_for(
        &elf,
        opts,
        bundle.num_epochs(),
        &bundle.touched_page_bases,
        bundle.num_private_input_pages,
    );
    // The same two lines `verify_global_bookends` builds its config from: each
    // AIR's main width, and the height the proof states.
    let shapes: Vec<(usize, usize)> = air_set
        .refs()
        .iter()
        .zip(&bundle.global.table_num_vars)
        .map(|(air, &num_vars)| (air.trace_layout().0, num_vars as usize))
        .collect();
    let config = crate::multilinear_prove::chain_config(&shapes);
    let plan = air_set.genesis_stack();
    let shape = (!plan.is_empty()).then_some(transcript_pin::StackShape {
        // Two preprocessed columns per dense page, which is what `at` holds.
        columns: plan.at.len(),
        // Every genesis page is one page tall, so the stack is a rectangle.
        num_vars: crate::continuation::PAGE_NUM_VARS,
    });
    transcript_pin::Stack { shape, config }
}

/// Everything about a run the pins are evaluated at, read off its bundle.
#[cfg(feature = "hash-metrics")]
fn run_facts(
    elf_bytes: &[u8],
    bundle: &crate::multilinear_continuation::ContinuationProof,
    opts: &crate::ProofOptions,
) -> RunFacts {
    let stack_chains = bundle
        .global
        .proof
        .preprocessed
        .iter()
        .flat_map(|opening| &opening.polys)
        .map(|poly| StackChain {
            folds: poly.rounds.iter().map(|r| r.sumcheck.len()).collect(),
            queries: poly
                .rounds
                .first()
                .map(|r| match &r.openings {
                    multilinear::whir_chain::RoundOpenings::Base(p) => p.current.len(),
                    multilinear::whir_chain::RoundOpenings::Extension(p) => p.current.len(),
                })
                .expect("a chain has at least one round"),
        })
        .collect();
    RunFacts {
        epochs: bundle.num_epochs() as u64,
        carried_roots: bundle
            .epochs
            .iter()
            .map(|e| e.proof.roots.len() as u64)
            .sum(),
        stack_chains,
        stack: global_stack(elf_bytes, bundle, opts),
    }
}

/// Asserts the pinned pair, or says out loud why it did not.
#[cfg(feature = "hash-metrics")]
fn check_transcript_pins(
    elf: &[u8],
    epoch_size_log2: u32,
    run: Option<&RunFacts>,
    prove: &crypto::hash_metrics::Counts,
    verify: &crypto::hash_metrics::Counts,
) {
    use sha2::{Digest, Sha256};

    let sha: String = Sha256::digest(elf)
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect();
    // ⚠ THE SAME READER THE PROVER USES, not a second parse of the same
    // variable: `MaxRowsConfig::default` is what chunked the epochs whose
    // transcript this is, and it reaches the posture through this function.
    let max_rows_log2 = crate::tables::max_rows_log2_override();
    if !pin_applies(&sha, elf.len(), epoch_size_log2, max_rows_log2) {
        // Never silent. A skipped assert that prints nothing is
        // indistinguishable from one that passed, which is the failure this
        // whole pin exists against.
        println!(
            "{}",
            pin_skip_line(&sha, elf.len(), epoch_size_log2, max_rows_log2)
        );
        return;
    }

    // ⚠ PAST THE GUARD WITH NO PROOF IS A DEFECT, NOT A DEGENERATE CASE. See
    // [`RunFacts`]: substituting zeros here would make the stack term vanish on
    // a run that has a stack, and the pin would then pass by describing a
    // different run.
    let run = run.expect(
        "the transcript pin applies to this run, but the bundle its derived \
         terms are evaluated at was not supplied",
    );
    let stack = &run.stack;

    // ★ THE EPOCH COUNT IS THE RUN'S, and the pin multiplies three terms by it.
    // Asserted rather than assumed, because a guest that produced fourteen
    // epochs would otherwise move every derived term at once and arrive as "the
    // counts moved".
    assert_eq!(
        run.epochs,
        transcript_pin::EPOCHS,
        "the pinned guest proved {} epochs at 2^{epoch_size_log2}; the pin was \
         derived at {}",
        run.epochs,
        transcript_pin::EPOCHS,
    );

    // ★★ `owed`'s CARRIED HALF, AGAINST THE RUN'S OWN PROOF. An epoch's carried
    // roots are one per stacked polynomial of every commitment group, and that
    // list is `EpochProof::proof.roots`. Summing it is the same quantity the
    // constant states, taken from the object the measurement came out of — so a
    // posture that moved the chain count reddens by this name instead of as an
    // unexplained delta on both pinned lines.
    assert_eq!(
        run.carried_roots,
        transcript_pin::OWED_CARRIED_ABSORBS,
        "the bundle's epochs carry {} roots between them - one per chain - and \
         `owed`'s carried half is pinned at {}; the chain count moved, which is \
         the table-chunking posture moving",
        run.carried_roots,
        transcript_pin::OWED_CARRIED_ABSORBS,
    );

    // ★★ AND THE GENESIS STACK THE TERMS WERE DERIVED AT IS THE ONE THE PROOF
    // CARRIES. `stack` comes from the verifier's own plan; this says the
    // cross-epoch proof actually contains the opening that plan implies, and
    // with the round structure the terms were priced at. Without it the stack
    // term would be a statement about a plan rather than about this run.
    assert_stack_matches_proof(stack, &run.stack_chains);

    // ★ THE SHAPE IS DERIVED FROM THIS GUEST, NOT ASSUMED. Every term the
    // prepared opening contributes is a function of the DECODE group's shape,
    // so the shape is read off the ELF here and asserted before a single count
    // is compared. Without it the pin could describe a program whose
    // instruction table changed size, and the disagreement would arrive as "the
    // counts moved" — the one message that says nothing about which of the two
    // is wrong.
    //
    // It costs the instruction map and no commitment, and it runs after both
    // measurement windows are closed.
    let shape = decode_prepared_shape(elf);
    assert_eq!(
        shape,
        transcript_pin::DECODE_PREPARED_SHAPE,
        "the pinned guest's DECODE group is {shape:?} columns/log2-rows; the pin \
         was derived at {:?}",
        transcript_pin::DECODE_PREPARED_SHAPE
    );

    let triple = |c: &crypto::hash_metrics::Counts| {
        (
            c.transcript_absorbs,
            c.transcript_squeezes,
            c.transcript_states,
        )
    };
    assert_pinned_pair(shape, stack, triple(prove), triple(verify));
    println!(
        "{:<12} transcript pin OK (both sides, and the owed delta)",
        "WHIR"
    );
}

/// The genesis stack the pin priced against the one the proof carries.
///
/// ⛔ THIS IS WHAT KEEPS THE STACK TERM A CHECK. Its counts come out of
/// `config.schedule(n_stack)` over a plan the VERIFIER derives from the ELF; if
/// the cross-epoch proof held a different opening — a chain of a different
/// length, a different number of stacked polynomials, a different fold
/// schedule, a different query count — the terms would be arithmetic about
/// something that did not happen and the pinned totals would still be whatever
/// they were. Every quantity the terms use is read back off the cross-epoch
/// `MultiProof` — in [`run_facts`], while the bundle is still alive — and
/// compared here.
///
/// ⚠ `Q` is read from a round's opening block rather than from the config,
/// because that is the number `whir_chain::verify` itself checks
/// (`openings.current.len() != config.num_queries`) and therefore the number
/// the transcript's query draws were made at.
#[cfg(feature = "hash-metrics")]
fn assert_stack_matches_proof(stack: &transcript_pin::Stack, chains: &[StackChain]) {
    let Some(shape) = stack.shape else {
        assert!(
            chains.is_empty(),
            "the run's genesis plan stacks no page, but the cross-epoch proof \
             carries a prepared opening of {} chains",
            chains.len(),
        );
        return;
    };
    assert!(
        !chains.is_empty(),
        "the run's genesis plan stacks {} columns, but the cross-epoch proof \
         carries no prepared opening",
        shape.columns,
    );

    let layout = transcript_pin::genesis_stack_layout(shape);
    let schedule = stack.config.schedule(layout.n_stack());
    assert_eq!(
        chains.len(),
        layout.num_polys(),
        "the genesis stack's layout is {} stacked polynomials at n_stack {}; \
         the cross-epoch opening carries {}",
        layout.num_polys(),
        layout.n_stack(),
        chains.len(),
    );
    for (i, chain) in chains.iter().enumerate() {
        assert_eq!(
            chain.folds,
            schedule,
            "the genesis stack's chain {i} folds {:?}; n_stack {} at folding {} \
             schedules {schedule:?}",
            chain.folds,
            layout.n_stack(),
            stack.config.log_folding,
        );
        assert_eq!(
            chain.queries, stack.config.num_queries,
            "the genesis stack's chain {i} opens {} positions a round; the \
             cross-epoch config draws {}",
            chain.queries, stack.config.num_queries,
        );
    }
}

/// The DECODE group's shape — columns, and log2 of the rows — as the ELF
/// implies it.
///
/// The two numbers `decode_prepared_for` derives before it commits anything,
/// taken the same way: `preprocessed_columns_from_elf` builds the instruction
/// table's columns, and the group is as wide as that list and as tall as one of
/// them.
#[cfg(feature = "hash-metrics")]
fn decode_prepared_shape(elf_bytes: &[u8]) -> (usize, usize) {
    let elf = Elf::load(elf_bytes).expect("the pinned guest loads");
    let columns =
        crate::tables::decode::preprocessed_columns_from_elf(&elf).expect("DECODE's columns");
    let rows = columns.first().expect("DECODE has columns").len();
    assert!(
        rows.is_power_of_two(),
        "DECODE's preprocessed columns are {rows} rows, which the hypercube cannot hold"
    );
    (columns.len(), rows.trailing_zeros() as usize)
}

/// The assertions themselves, split from the guard so they can be reached
/// without the pinned guest.
///
/// ⚠ This split is the point, not tidiness. With the guard and the assertions
/// in one function, the assertions ran on the box and NOWHERE ELSE: a mistyped
/// constant, a swapped pair or a broken delta would have been discovered by a
/// GPU run rather than by `cargo test`. Taking the triples as arguments makes
/// every branch reachable from a laptop, which is why the four tests below
/// exist and why three of them are `should_panic`.
#[cfg(feature = "hash-metrics")]
fn assert_pinned_pair(
    shape: (usize, usize),
    stack: &transcript_pin::Stack,
    prove: (u64, u64, u64),
    verify: (u64, u64, u64),
) {
    // Only the state columns are destructured: the two lines are compared whole
    // against their pins, and the delta between them is a property of the
    // CONSTANTS rather than of a measurement — see
    // `the_pinned_constants_differ_by_owed`.
    let (_, _, pt) = prove;
    let (_, _, vt) = verify;

    assert_eq!(
        prove,
        transcript_pin::prove(shape, stack),
        "the PROVE-side transcript counts moved"
    );
    assert_eq!(
        verify,
        transcript_pin::verify(shape, stack),
        "the VERIFY-side transcript counts moved"
    );

    // ⚠ NO `owed` ASSERTION HERE, and its absence is deliberate. Once both
    // lines match their pins, their difference is forced — `OWED` is
    // `VERIFY - PROVE` by construction, so a third runtime assertion could
    // never fire. Writing its test is what exposed that: the constructed
    // counter-example was rejected by the VERIFY assertion two lines up,
    // never reaching the delta.
    //
    // The delta is a statement about the CONSTANTS, not about a measurement,
    // so it is checked where it can fail — see
    // [`the_pinned_constants_differ_by_owed`].

    // The control that costs nothing: one `state()` per grind check, so this
    // column and the grind count are two instruments on one quantity.
    assert_eq!(
        pt, vt,
        "the two sides disagree on state reads, which are grind checks on both"
    );
}

/// ★★ The constants are the measurement — asserted against LITERALS.
///
/// Passing `transcript_pin::prove()` here would be a check that cannot fail: a
/// mutated constant would move the input and the expectation together and the
/// test would pass on any value. The numbers below are written out so that a
/// constant which drifts, is mistyped, or has its two lines swapped fails here,
/// on a laptop, rather than on the box an hour later.
///
/// ⚠ The literals are the BASES, and the per-branch term is recomputed from the
/// struct — because that term is what differs between the seam lineage and this
/// merged one. Writing the totals out would make this test the thing that has
/// to be edited on every branch, which is precisely the property the split
/// removed from the pin.
#[cfg(feature = "hash-metrics")]
#[test]
fn the_pinned_pair_is_the_measurement() {
    let counts = 15 * crate::statement::NUM_TABLE_KINDS as u64;
    // The derived-root term, re-spelled from the protocol rather than called
    // from `transcript_pin` — a re-derivation that calls the thing it checks is
    // not one. One root per epoch into the prover's roots block; two per epoch
    // on the verify side, because the `owed` replay absorbs it as well.
    //
    // ★ AND THE PREPARED OPENING'S TERM, re-spelled the same way. The pin
    // derives it by calling `schedule()`; this writes the arithmetic out at the
    // shape that schedule implies — six rounds over twenty-three variables at
    // fold width four, 112 queries, four candidates a squeeze — so the two
    // spellings disagree if either the round structure or the shape moves.
    // `the_prepared_opening_is_the_schedule_the_shape_implies` is what ties
    // those four numbers to the shape rather than to this comment.
    let (rounds, folded, queries) = (6u64, 23u64, 112u64);
    let grinds = 3 * rounds - 1;
    let opening = (
        5 + grinds + 2 * folded + 2 * (rounds - 1) + 1,
        1 + folded + 2 * (rounds - 1) + rounds * queries.div_ceil(4),
        grinds,
    );

    // The cross-epoch proof's chain posture, through the one function that
    // decides it. `one_stack(n, 1)` is `n`, so a one-column shape pins the
    // tallest stacked polynomial exactly — and the query count is 112 for every
    // height the block's cross-epoch tables can reach. The RUNTIME pin does not
    // rely on that: it evaluates the terms at the run's own config.
    let config = crate::multilinear_prove::chain_config(&[(1, 21)]);
    assert_eq!(
        (config.log_folding, config.num_queries),
        (4, 112),
        "the cross-epoch chain posture moved; every stack term below is priced \
         at folding 4 and 112 queries"
    );

    // ★ THE STACK-LESS PAIR FIRST — the run the bases were measured at (lb11,
    // whose genesis was entirely sparse, so its cross-epoch proof carried no
    // prepared opening at all). A `None` stack has to reproduce those totals
    // exactly, or the stack term is a re-baseline wearing an addition's clothes.
    let sparse = transcript_pin::Stack {
        shape: None,
        config,
    };
    assert_eq!(
        sparse.terms(),
        (0, 0, 0),
        "a run with no dense genesis page must cost the transcript nothing; \
         `genesis_prepared_for` returns `None` and the proof is byte for byte \
         the one it was before this route existed"
    );
    assert_pinned_pair(
        transcript_pin::DECODE_PREPARED_SHAPE,
        &sparse,
        (
            405_777 + counts + 15 + 15 * opening.0,
            146_091 + 15 * opening.1,
            3_156 + 15 * opening.2,
        ),
        (
            405_922 + counts + 30 + 15 * opening.0,
            146_121 + 15 * opening.1,
            3_156 + 15 * opening.2,
        ),
    );

    // ★★ THEN THE BLOCK'S PAIR, whose genesis stacks three dense pages: two
    // preprocessed columns each, six columns of 2^18, one stacked polynomial at
    // n_stack 21, six rounds at fold width four. Written out as the arithmetic
    // — the stack's roots in the cross-epoch roots block, one claimed value per
    // stacked column and one batching challenge, then the chain — so that a
    // schedule or a layout that moved disagrees with this spelling rather than
    // moving both sides of the comparison together.
    // The literal 18 below is a page's height; bound here so it cannot drift
    // away from the constant the prover stacks at.
    assert_eq!(crate::continuation::PAGE_NUM_VARS, 18);
    let (s_rounds, s_folded) = (6u64, 21u64);
    let s_grinds = 3 * s_rounds - 1;
    let stack = (
        1 + 6 + s_grinds + 2 * s_folded + 2 * (s_rounds - 1) + 1,
        1 + s_folded + 2 * (s_rounds - 1) + s_rounds * queries.div_ceil(4),
        s_grinds,
    );
    let block = transcript_pin::Stack {
        shape: Some(transcript_pin::StackShape {
            columns: 6,
            num_vars: 18,
        }),
        config,
    };
    assert_eq!(
        block.terms(),
        stack,
        "the block's genesis stack priced from `schedule()` disagrees with the \
         same arithmetic written out"
    );
    assert_pinned_pair(
        transcript_pin::DECODE_PREPARED_SHAPE,
        &block,
        (
            405_777 + counts + 15 + 15 * opening.0 + stack.0,
            146_091 + 15 * opening.1 + stack.1,
            3_156 + 15 * opening.2 + stack.2,
        ),
        (
            405_922 + counts + 30 + 15 * opening.0 + stack.0,
            146_121 + 15 * opening.1 + stack.1,
            3_156 + 15 * opening.2 + stack.2,
        ),
    );
}

/// ★★ THE STACK'S TERMS ARE THE SCHEDULE ITS SHAPE IMPLIES, at both shapes the
/// tree can reach — and the expectations are ARITHMETIC, not results.
///
/// ⛔ WHY TWO SHAPES AND NOT ONE. A form pinned at a single shape is a constant
/// with extra steps: every term could be wrong in a way that happens to sum
/// right there. The two shapes differ in the one input the whole derivation
/// hangs off — `one_stack(18, columns)`, which is 19 for a lone dense page and
/// 21 for the block's three — and they differ in the ROUND COUNT that falls out
/// of it, so a mistake in the round structure cannot be flat across both.
///
/// The fixture's dense guest (`dense_data_page_touch`) is the 19; the block's
/// plan is the 21.
#[cfg(feature = "hash-metrics")]
#[test]
fn the_genesis_stack_is_the_schedule_the_shape_implies() {
    // The cross-epoch posture, through the one function that decides it —
    // `one_stack(n, 1)` is `n`, so a one-column shape pins the tallest stacked
    // polynomial exactly. Stated here because the literal triple at the end of
    // this test is only the block's numbers at THIS posture; the runtime pin
    // evaluates the same form at the run's own config and does not rely on it.
    let config = crate::multilinear_prove::chain_config(&[(1, 21)]);
    assert_eq!(
        (config.log_blowup, config.log_folding, config.num_queries),
        (2, 4, 112),
        "the cross-epoch chain posture moved"
    );
    let per_round_queries = (config.num_queries as u64).div_ceil(4);

    // A shape, and the terms spelled from it rather than from `schedule()`.
    let expect = |columns: usize, n_stack: u64, rounds: u64| {
        let grinds = 3 * rounds - 1;
        (
            // the stack's one root, absorbed in the cross-epoch roots block
            1
                // one claimed value per stacked column
                + columns as u64
                // the chain: nonces, sumcheck evaluations, successor roots and
                // out-of-domain values on the non-final rounds, and the final
                // folded constant
                + grinds
                + 2 * n_stack
                + 2 * (rounds - 1)
                + 1,
            // the batching challenge, then the chain's own draws
            1 + n_stack + 2 * (rounds - 1) + rounds * per_round_queries,
            grinds,
        )
    };

    for (columns, n_stack, rounds) in [(2usize, 19u64, 5u64), (6, 21, 6)] {
        let shape = transcript_pin::StackShape {
            columns,
            num_vars: crate::continuation::PAGE_NUM_VARS,
        };
        let layout = transcript_pin::genesis_stack_layout(shape);
        assert_eq!(
            (layout.n_stack() as u64, layout.num_polys()),
            (n_stack, 1),
            "{columns} columns of 2^{} stack into one polynomial at {n_stack} \
             variables",
            crate::continuation::PAGE_NUM_VARS,
        );
        assert_eq!(
            config.schedule(layout.n_stack()).len() as u64,
            rounds,
            "n_stack {n_stack} at fold width {} is {rounds} rounds",
            config.log_folding,
        );
        assert_eq!(
            transcript_pin::genesis_stack_terms(Some(shape), &config),
            expect(columns, n_stack, rounds),
            "the stack's terms at {columns} columns / n_stack {n_stack}"
        );
        // The card's side of the same shape: the held commitment plus the
        // chain's successor codewords, which is one commit per round.
        assert_eq!(
            transcript_pin::genesis_stack_commits(Some(shape), &config),
            rounds,
            "the stack costs one device commit per chain round at n_stack \
             {n_stack}"
        );
    }

    // ★ AND THE BLOCK'S NUMBERS, stated once so the measurement they explain is
    // legible here: runs lb17 and lb18 read +77 absorbs, +200 squeezes and +17
    // states on BOTH sides against a pin taken before the stack existed.
    let block = transcript_pin::StackShape {
        columns: 6,
        num_vars: crate::continuation::PAGE_NUM_VARS,
    };
    assert_eq!(
        transcript_pin::genesis_stack_terms(Some(block), &config),
        (77, 200, 17)
    );

    // Nothing stacked, nothing owed.
    assert_eq!(
        transcript_pin::genesis_stack_terms(None, &config),
        (0, 0, 0)
    );
    assert_eq!(transcript_pin::genesis_stack_commits(None, &config), 0);
}

/// The shape DERIVER, exercised on a guest a laptop has.
///
/// ⚠ `check_transcript_pins` reaches its shape assertion only behind the sha
/// guard, so on the box and nowhere else — the same blindness the pin's own
/// split was written to remove. The assertion compares a derived tuple against
/// a constant, and the way that comparison goes wrong without the guest is the
/// tuple: two `usize`s, and nothing in the type says which is the width. So the
/// deriver runs here on `sub`, where the column count is a constant of the
/// DECODE table and the height is whatever that program implies.
#[cfg(feature = "hash-metrics")]
#[test]
fn the_shape_deriver_reads_columns_then_log2_rows() {
    let elf_bytes = crate::test_utils::asm_elf_bytes("sub");
    let elf = Elf::load(&elf_bytes).expect("load");
    let built =
        crate::tables::decode::preprocessed_columns_from_elf(&elf).expect("DECODE's columns");

    let (columns, num_vars) = decode_prepared_shape(&elf_bytes);
    assert_eq!(
        columns,
        built.len(),
        "the first element is the column count"
    );
    assert_eq!(
        1usize << num_vars,
        built[0].len(),
        "the second element is log2 of the rows"
    );
    assert_eq!(
        columns,
        crate::tables::decode::NUM_PRECOMPUTED_COLS,
        "DECODE's column count is the same for every guest; only the height moves"
    );
}

/// ★★ WHAT THE PINNED SHAPE IMPLIES, written where a laptop can falsify it.
///
/// Every term the prepared opening adds is a function of four numbers the shape
/// determines: the stacked width, the number of polynomials, the fold schedule
/// and the query count. They are asserted here, so a change to the stacking
/// rule, to `MAX_STACK_VARS`, to the fold width or to the shipped query count
/// fails on a laptop with a name attached instead of arriving on the box as a
/// pin that moved by an unexplained amount.
///
/// The query count is not a number invented here: `query_count` pins the
/// shipped posture's 110 / 112 / 113 in its own tests, and this says which of
/// them this shape lands on.
#[cfg(feature = "hash-metrics")]
#[test]
fn the_prepared_opening_is_the_schedule_the_shape_implies() {
    let shape = transcript_pin::DECODE_PREPARED_SHAPE;
    let (columns, _) = shape;
    let (layout, config) = transcript_pin::prepared_group(shape);

    // Five columns of 2^20 is 5,242,880 cells, which the stacker rounds up to
    // 2^23 — one polynomial, so one chain and one root.
    assert_eq!(layout.n_stack(), 23, "the DECODE group's stacked width");
    assert_eq!(layout.num_polys(), 1, "the DECODE group is one polynomial");
    assert_eq!(
        layout.placements().len(),
        columns,
        "one placement per column, which is what the opening's wrapper absorbs"
    );
    assert_eq!(
        config.schedule(layout.n_stack()),
        vec![4, 4, 4, 4, 4, 3],
        "the fold schedule the chain runs"
    );
    assert_eq!(
        config.num_queries, 112,
        "the shipped query count at this shape"
    );

    // …and the terms those four numbers produce.
    assert_eq!(
        transcript_pin::prepared_opening_per_epoch(shape),
        (79, 202, 17),
        "the opening's per-epoch transcript cost"
    );
    assert_eq!(
        transcript_pin::prepared_fold_commits_per_epoch(shape),
        5,
        "one successor codeword per non-final round"
    );
    assert_eq!(
        transcript_pin::prepared_commitments(shape),
        1,
        "the commitment itself, once for the run"
    );
}

/// The genesis stack the pinned run carries, for the tests below that need one
/// as an INPUT rather than as an expectation.
///
/// ⚠ Never used by [`the_pinned_pair_is_the_measurement`], which spells the
/// stack's arithmetic out instead: a test whose expectation comes from the same
/// call as its subject is the check that cannot fail.
#[cfg(feature = "hash-metrics")]
fn pinned_stack() -> transcript_pin::Stack {
    transcript_pin::Stack {
        shape: Some(transcript_pin::StackShape {
            columns: 6,
            num_vars: crate::continuation::PAGE_NUM_VARS,
        }),
        config: crate::multilinear_prove::chain_config(&[(1, 21)]),
    }
}

/// ★ One unit on the prove line fails on the prove assertion.
#[cfg(feature = "hash-metrics")]
#[test]
#[should_panic(expected = "the PROVE-side transcript counts moved")]
fn a_prove_count_off_by_one_is_rejected() {
    let shape = transcript_pin::DECODE_PREPARED_SHAPE;
    let stack = pinned_stack();
    let (a, s, t) = transcript_pin::prove(shape, &stack);
    assert_pinned_pair(
        shape,
        &stack,
        (a + 1, s, t),
        transcript_pin::verify(shape, &stack),
    );
}

/// ★ One unit on the verify line fails on the verify assertion.
#[cfg(feature = "hash-metrics")]
#[test]
#[should_panic(expected = "the VERIFY-side transcript counts moved")]
fn a_verify_count_off_by_one_is_rejected() {
    let shape = transcript_pin::DECODE_PREPARED_SHAPE;
    let stack = pinned_stack();
    let (a, s, t) = transcript_pin::verify(shape, &stack);
    assert_pinned_pair(
        shape,
        &stack,
        transcript_pin::prove(shape, &stack),
        (a + 1, s, t),
    );
}

/// ★★ The per-branch term is a TERM, not a re-baseline.
///
/// The pin's absorb totals are `base + EPOCHS * NUM_TABLE_KINDS`, and this says
/// the split is the one the protocol makes: the kind count is the length of the
/// list `absorb_table_counts` walks, and an epoch statement absorbs exactly that
/// many counts.
///
/// It can fail. `NUM_TABLE_KINDS` is checked against the array
/// `statement::table_count_values` actually returns — which is the one place the
/// destructure of `TableCounts` is written — so a field added to `TableCounts`
/// and pushed into the array without bumping the constant fails to compile, and
/// a constant bumped without the field fails here.
#[cfg(feature = "hash-metrics")]
#[test]
fn the_per_branch_term_is_the_table_kind_count() {
    let zero = crate::TableCounts {
        cpu: 0,
        lt: 0,
        memw: 0,
        memw_aligned: 0,
        load: 0,
        mul: 0,
        dvrm: 0,
        shift: 0,
        branch: 0,
        memw_register: 0,
        eq: 0,
        bytewise: 0,
        store: 0,
        cpu32: 0,
        keccak: 0,
        keccak_rnd: 0,
        ecsm: 0,
        ecdas: 0,
        hint: 0,
        commit: 0,
        blake3: 0,
    };
    let kinds = crate::statement::table_count_values(&zero).len();
    assert_eq!(
        kinds,
        crate::statement::NUM_TABLE_KINDS,
        "NUM_TABLE_KINDS is not the number of counts a statement absorbs"
    );
    assert_eq!(
        transcript_pin::table_count_absorbs(),
        transcript_pin::EPOCHS * kinds as u64,
        "the pin's per-branch term is not `epochs x kinds`"
    );
    let shape = transcript_pin::DECODE_PREPARED_SHAPE;
    let stack = pinned_stack();
    let opening_absorbs =
        transcript_pin::EPOCHS * transcript_pin::prepared_opening_per_epoch(shape).0;
    assert_eq!(
        transcript_pin::prove(shape, &stack).0 - transcript_pin::PROVE_BASE_ABSORBS,
        transcript_pin::table_count_absorbs()
            + transcript_pin::prove_derived_root_absorbs()
            + opening_absorbs
            + stack.terms().0,
    );
    assert_eq!(
        transcript_pin::verify(shape, &stack).0 - transcript_pin::VERIFY_BASE_ABSORBS,
        transcript_pin::table_count_absorbs()
            + transcript_pin::verify_derived_root_absorbs()
            + opening_absorbs
            + stack.terms().0,
    );
}

/// ★★ THE DERIVED DELTA, checked where it can actually fail: on the constants.
///
/// `owed` is the only thing the verifier does that the prover does not, so the
/// two pinned lines must differ by exactly it — `Sum roots.len()` absorbs,
/// `2 x epochs` squeezes, no state reads. That is a claim about the pair of
/// constants, and it is the claim that survives a re-baseline: if someone
/// measures a new run and updates PROVE and VERIFY together, this fires unless
/// `owed` is still `owed`, forcing them to look at why the difference moved
/// rather than carrying a changed protocol into two numbers that agree with
/// each other.
///
/// ⚠ It lives here rather than inside [`assert_pinned_pair`] because there it
/// could never fire: with both lines asserted against their pins, their
/// difference is forced. That was found by writing the test — the constructed
/// counter-example never reached the delta, because the VERIFY assertion
/// rejected it first.
#[cfg(feature = "hash-metrics")]
#[test]
fn the_pinned_constants_differ_by_owed() {
    let shape = transcript_pin::DECODE_PREPARED_SHAPE;
    // ⚠ EVALUATED AT BOTH STACK POSTURES, because the claim is that the genesis
    // stack does NOT enter `owed`. It costs both sides the same amount — the
    // cross-epoch proof has no replay — so the difference between the two lines
    // has to be the same with a stack and without one. A term that leaked into
    // one line only would show up here and in no other test.
    for stack in [
        pinned_stack(),
        transcript_pin::Stack {
            shape: None,
            config: crate::multilinear_prove::chain_config(&[(1, 21)]),
        },
    ] {
        let (pa, ps, pt) = transcript_pin::prove(shape, &stack);
        let (va, vs, vt) = transcript_pin::verify(shape, &stack);
        assert_eq!(
            (va - pa, vs - ps, vt - pt),
            transcript_pin::OWED,
            "the two pinned lines no longer differ by `owed` — one was \
             re-baselined without the other, or the protocol changed"
        );
    }

    // …and `owed` is itself derived, not observed: the absorbs are one per root
    // over the 15 epoch calls, 30 squeezes is two per call. Stating the shape
    // means a future epoch count cannot silently keep the old constant.
    let (oa, os, ot) = transcript_pin::OWED;
    assert_eq!(
        os,
        2 * transcript_pin::EPOCHS,
        "`owed` samples twice per epoch call"
    );
    assert_eq!(ot, 0, "`owed` reads no transcript state");
    // ★ The out-of-band opening's own term, which IS derivable here: the replay
    // absorbs DECODE's derived root once per epoch on top of the carried ones.
    // Without this the two halves of the wiring — the root reaching
    // `multi_verify` and the root reaching the replay — could be re-pinned one
    // at a time, which is exactly the drift that made this branch red.
    assert_eq!(
        oa - transcript_pin::OWED_CARRIED_ABSORBS,
        transcript_pin::prove_derived_root_absorbs(),
        "`owed` no longer absorbs DECODE's derived root once per epoch"
    );
    // ⚠ Only the DERIVED half of the absorb count is asserted here, and that is
    // a limit of this test rather than of the pin. The carried half is
    // `Sum roots.len()` over the epochs — data from the table shapes at this
    // posture, which no form in this tree computes — so any predicate written
    // about it HERE would be either circular (comparing the constant to itself)
    // or vacuous. An earlier draft had `oa % 1 == 0`, which is true of every
    // integer. It is checked where it can fail instead: `check_transcript_pins`
    // sums the bundle's own epoch roots against it, on the run the constant
    // describes.
}

/// ⛔ THE TABLE CAP IS A MEASUREMENT, not a role the identity merely has a slot
/// for.
///
/// A pin whose identity carried the cap but whose VALUE drifted would skip on
/// the record posture and apply to some other one, silently — the same failure
/// as the fabricated sha tail, one field over.
#[cfg(feature = "hash-metrics")]
#[test]
fn the_pinned_table_cap_is_the_record_posture() {
    assert_eq!(
        transcript_pin::MAX_ROWS_LOG2,
        21,
        "the record posture is a uniform 2^21 cap: the WHIR tree, the D-S \
         control and the block artifact all export LAMBDA_VM_MAX_ROWS_LOG2=21, \
         and these constants were measured there"
    );
}

/// ★ The guard REFUSES every posture but the pinned one, including the cap-less
/// default the old constants were measured at.
///
/// ⛔ EXECUTED ON A STATE IT MUST ACCEPT AND ON STATES IT MUST REFUSE. The
/// accepting arm is what makes the refusals mean something: a guard that
/// refused everything would pass a test that only tried wrong values.
#[cfg(feature = "hash-metrics")]
#[test]
fn the_pin_applies_only_at_the_posture_it_was_measured_at() {
    let pinned = Some(transcript_pin::MAX_ROWS_LOG2);
    assert!(
        pin_applies(
            transcript_pin::ELF_SHA256,
            transcript_pin::ELF_LEN,
            transcript_pin::EPOCH_LOG2,
            pinned,
        ),
        "the pinned identity must be accepted by its own guard"
    );
    // `None` is the production per-table caps — a real posture, and the one
    // that made this pin fail while looking like it applied.
    for cap in [
        None,
        Some(transcript_pin::MAX_ROWS_LOG2 - 1),
        Some(transcript_pin::MAX_ROWS_LOG2 + 1),
    ] {
        assert!(
            !pin_applies(
                transcript_pin::ELF_SHA256,
                transcript_pin::ELF_LEN,
                transcript_pin::EPOCH_LOG2,
                cap,
            ),
            "a run at table cap {} is not the pinned configuration",
            cap_label(cap),
        );
    }
    // And the skip line names BOTH caps, or a reader cannot tell which posture
    // they ran at.
    let line = pin_skip_line(
        transcript_pin::ELF_SHA256,
        transcript_pin::ELF_LEN,
        transcript_pin::EPOCH_LOG2,
        None,
    );
    assert!(
        line.contains("UNSET (the production per-table caps)")
            && line.contains(&format!("uniform 2^{}", transcript_pin::MAX_ROWS_LOG2)),
        "the skip line must name the run's cap and the pinned one: {line}"
    );
}

/// ★★ A sha that agrees on a PREFIX is refused, and the skip line shows why.
///
/// The defect this is written against: the constant's last 48 hex were
/// fabricated, so the guard skipped on the pinned guest itself — and the skip
/// line printed 16 characters of each side, the exact width at which the real
/// sha and the invented one agreed. The failure was invisible in the output of
/// the thing that existed to report it.
///
/// Both halves are needed. The refusal alone would pass with a truncated
/// diagnostic; the diagnostic alone would pass with a comparison that only
/// looked at a prefix.
#[cfg(feature = "hash-metrics")]
#[test]
fn a_sha_agreeing_only_on_the_prefix_is_refused_and_says_so() {
    // Same first 16 hex, different tail — one `format!`, no ELF to forge.
    let near_miss = format!("{}{}", &transcript_pin::ELF_SHA256[..16], "0".repeat(48));
    assert_eq!(near_miss.len(), 64);
    assert_eq!(
        near_miss[..16],
        transcript_pin::ELF_SHA256[..16],
        "the near miss must agree on the prefix, or it tests nothing"
    );
    assert_ne!(near_miss, transcript_pin::ELF_SHA256);

    assert!(
        !pin_applies(
            &near_miss,
            transcript_pin::ELF_LEN,
            transcript_pin::EPOCH_LOG2,
            Some(transcript_pin::MAX_ROWS_LOG2),
        ),
        "a sha differing only after position 16 was accepted: the comparison is \
         looking at a prefix"
    );

    // …and the line it prints must make the difference visible.
    let line = pin_skip_line(
        &near_miss,
        transcript_pin::ELF_LEN,
        transcript_pin::EPOCH_LOG2,
        Some(transcript_pin::MAX_ROWS_LOG2),
    );
    assert!(
        line.contains(&near_miss),
        "the skip line does not carry the found sha in full: {line}"
    );
    assert!(
        line.contains(transcript_pin::ELF_SHA256),
        "the skip line does not carry the pinned sha in full: {line}"
    );

    // The property in one assertion: whatever the line shows of each side, the
    // two shown values must differ. A prefix display fails here.
    let shown: Vec<&str> = line.split_whitespace().filter(|w| w.len() == 64).collect();
    assert_eq!(
        shown.len(),
        2,
        "expected two 64-hex values in the skip line, found {}: {line}",
        shown.len()
    );
    assert_ne!(
        shown[0], shown[1],
        "the skip line shows the same value twice for a genuine mismatch"
    );
}

/// ★ The pinned constant is a full-width sha, not a truncation.
#[cfg(feature = "hash-metrics")]
#[test]
fn the_pinned_sha_is_full_width() {
    assert_eq!(
        transcript_pin::ELF_SHA256.len(),
        64,
        "a sha256 is 64 hex characters; anything shorter is a display that got \
         pinned"
    );
    assert!(
        transcript_pin::ELF_SHA256
            .bytes()
            .all(|b| b.is_ascii_hexdigit() && !b.is_ascii_uppercase()),
        "the pinned sha is not lowercase hex"
    );
}

/// ★ The guard skips rather than fires on a guest that is not the pinned one.
///
/// Card-free and not `#[ignore]`d, because the skip path is the half that runs
/// on every other invocation of the bench and the half that would fail silently
/// if it were wrong. If the guard were inverted — asserting on the wrong ELF —
/// every run on any other program would panic on counts that were never about
/// it; if it were absent, the pin would be decorative.
///
/// The counts passed in are deliberately absurd. Reaching the assertions with
/// them would panic, so a test that returns at all proves the guard returned
/// first.
#[cfg(feature = "hash-metrics")]
#[test]
fn the_transcript_pin_skips_a_guest_it_does_not_recognise() {
    let nonsense = crypto::hash_metrics::Counts {
        transcript_absorbs: 1,
        transcript_squeezes: 2,
        transcript_states: 3,
        ..Default::default()
    };

    // ⚠ `None` FOR THE RUN'S FACTS, and that strengthens this test rather than
    // weakening it: past the guard the pin panics on a `None`, so a call that
    // returns proves the guard returned first on the identity alone.
    // Wrong bytes, wrong length.
    check_transcript_pins(
        b"not an elf",
        transcript_pin::EPOCH_LOG2,
        None,
        &nonsense,
        &nonsense,
    );

    // ⚠ Right length, wrong bytes — the guard must be the sha and not the size,
    // which is the whole point of preferring it to the ELF's name.
    let same_length = vec![0u8; transcript_pin::ELF_LEN];
    check_transcript_pins(
        &same_length,
        transcript_pin::EPOCH_LOG2,
        None,
        &nonsense,
        &nonsense,
    );
}

#[test]
#[ignore]
fn continuations() {
    let name = std::env::var("LAMBDA_VM_BENCH_ELF").unwrap_or_else(|_| "ethrex".into());
    let input = std::env::var("LAMBDA_VM_BENCH_INPUT").unwrap_or_default();
    let backend = std::env::var("LAMBDA_VM_BENCH_BACKEND").unwrap_or_else(|_| "both".into());
    let epoch_size_log2: u32 = std::env::var("LAMBDA_VM_BENCH_EPOCH_LOG2")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(20);
    let bytes = elf_bytes(&name);
    let inputs = input_bytes(&input);
    let opts = options();
    let threads = std::env::var("RAYON_NUM_THREADS").unwrap_or_else(|_| "all".into());
    let label = if input.is_empty() { &name } else { &input };
    println!(
        "\n{label} — continuation, epoch_size_log2={epoch_size_log2}, \
RAYON_NUM_THREADS={threads}, backend={backend}"
    );

    let mib = |n: usize| n as f64 / (1024.0 * 1024.0);
    let mut fri = None;
    let mut whir = None;
    // ★ THE COUNTS ARE TAKEN IN THE ARM'S OWN WINDOW; ONLY THE COMPARISON IS
    // DEFERRED, to below the timing table.
    //
    // Run a2v's pin went red and the run printed no prove time, no verify time
    // and no proof size: the assert panicked before the table, so a wrong
    // prediction cost the measurement that would have helped explain it. A pin
    // is a gate on a measurement, and a gate that destroys its own subject is
    // worth one line of plumbing to avoid.
    #[cfg(feature = "hash-metrics")]
    let mut pinned: Option<(
        crypto::hash_metrics::Counts,
        crypto::hash_metrics::Counts,
        RunFacts,
    )> = None;
    // The device commit count, read where the read-0 line below reads it so the
    // two can never describe different windows.
    #[cfg(all(feature = "cuda", feature = "hash-metrics"))]
    let mut device_commits: Option<(u64, u64)> = None;

    if backend != "whir" {
        let start = Instant::now();
        let bundle =
            crate::continuation::prove_continuation(&bytes, &inputs, epoch_size_log2, &opts)
                .expect("univariate continuation");
        let prove = start.elapsed();
        let size = rkyv::to_bytes::<rkyv::rancor::Error>(&bundle)
            .expect("serialize")
            .len();
        let epochs = bundle.num_epochs();
        let start = Instant::now();
        let output = crate::continuation::verify_continuation(&bytes, &bundle, &opts)
            .expect("univariate verify");
        assert!(output.is_some(), "the univariate continuation must verify");
        fri = Some((prove, start.elapsed(), size, epochs));
    }

    if backend != "fri" {
        // Held from the prove window to the verify one so the pair - and the
        // `owed` delta between them - can be asserted together. Uninitialised
        // and assigned exactly once: an `Option` here would carry a `None` the
        // compiler can prove is never read, since the assignment dominates the
        // use and both sit in this one branch.
        #[cfg(feature = "hash-metrics")]
        let prove_counts;
        // ★ Zeroed per arm, so the counts below belong to THIS prove and not to
        // whatever ran before it in the process.
        #[cfg(feature = "cuda")]
        {
            crypto::grinding::reset_gpu_grind_calls();
            multilinear::gpu::reset_call_counters();
        }
        // ★ Same reason, for the transcript counters: the line printed below
        // must be THIS arm's and not the process's running total.
        #[cfg(feature = "hash-metrics")]
        crypto::hash_metrics::reset();
        let start = Instant::now();
        let bundle = crate::multilinear_continuation::prove_continuation(
            &bytes,
            &inputs,
            epoch_size_log2,
            &opts,
        )
        .expect("multilinear continuation");
        let prove = start.elapsed();

        // ★★ READ 0 FOR EVERY WHIR ARM — which dispatches actually reached the
        // card. The `★ WHIR HASH:` banner says which hash was SELECTED; these
        // say which kernels RAN, and the two are not the same claim.
        //
        // A measured RPX arm once came in 14.5x slower than keccak with a
        // correct, KAT-pinned grind kernel sitting unused, because the host-side
        // dispatch had no arm for it. Nothing in this bench's output named the
        // cause: prove time, verify time, proof size and epoch count were all
        // consistent with "the hash is just expensive". A grind count of ZERO
        // beside a commit count of thousands says it in one line.
        #[cfg(feature = "cuda")]
        println!(
            "{:<12} gpu commits {} · host fallbacks {} · keccak grinds {} · rpx grinds {}",
            "WHIR",
            multilinear::gpu::commit_calls(),
            multilinear::gpu::host_fallbacks(),
            crypto::grinding::gpu_grind_calls(),
            crypto::grinding::gpu_grind_calls_rpx(),
        );
        #[cfg(all(feature = "cuda", feature = "hash-metrics"))]
        {
            // The model beside the reading, taken from the bundle that produced
            // it so the two cannot describe different runs.
            //
            // ⚠ THE EPOCHS AND THE CROSS-EPOCH PROOF GO IN SEPARATELY, because
            // the two prepared openings they carry are held at different
            // scopes — DECODE's once per ELF, the genesis stack's once per
            // `prove_global` call. Flattening them into one list is what let
            // the stack's held commitment go uncounted.
            let epoch_proofs: Vec<&stark::multilinear_table::MultiProof<_, _>> =
                bundle.epochs.iter().map(|e| &e.proof).collect();
            device_commits = Some((
                multilinear::gpu::commit_calls(),
                transcript_pin::commits(&epoch_proofs, &bundle.global.proof),
            ));
        }
        // ★★ WHICH SPONGE THE TRANSCRIPT RAN ON, per arm and on BOTH sides.
        //
        // The line above says which KERNELS ran; this one says which sponge the
        // Fiat-Shamir transcript used, and they are not the same claim. For
        // four measured A/Bs the RPX arm ran an RPX Merkle backend, an RPX
        // grind and a KECCAK transcript, and nothing printed here disagreed.
        //
        // Both sides are printed because one is not evidence. A counter on the
        // RPX side alone reads "rpx > 0, keccak 0" — and that keccak zero is
        // equally true when the keccak transcript ran and nobody instrumented
        // it, which is exactly the state that hid. `unattributed` is printed
        // for the same reason one level out: a third configuration arriving
        // with no counter of its own would otherwise look like silence.
        #[cfg(feature = "hash-metrics")]
        {
            let c = crypto::hash_metrics::snapshot();
            print_transcript_counts("WHIR prove", &c);
            prove_counts = c;
        }
        let size = rkyv::to_bytes::<rkyv::rancor::Error>(&bundle)
            .expect("serialize")
            .len();
        let epochs = bundle.num_epochs();
        // ★★ The VERIFY side gets its own window, and it is the side that
        // matters for recursion: an LFM replays the VERIFIER's transcript, not
        // the prover's. The two are different numbers over different work —
        // differencing one against a closed form derived for the other is the
        // same category error as comparing a prove stopwatch to a verify one.
        #[cfg(feature = "hash-metrics")]
        crypto::hash_metrics::reset();
        let start = Instant::now();
        let ok = crate::multilinear_continuation::verify_continuation(&bytes, &bundle, &opts)
            .expect("multilinear verify");
        assert!(ok, "the multilinear continuation must verify");
        #[cfg(feature = "hash-metrics")]
        {
            let verify_counts = crypto::hash_metrics::snapshot();
            print_transcript_counts("WHIR verify", &verify_counts);
            // ★ READ HERE, WHERE THE BUNDLE STILL EXISTS. The pins themselves
            // run after the results table — a red one must not cost a
            // measurement — and the bundle does not live that long, so every
            // fact the derived terms need is taken off it now and carried out
            // as numbers.
            pinned = Some((
                prove_counts,
                verify_counts,
                run_facts(&bytes, &bundle, &opts),
            ));
        }
        whir = Some((prove, start.elapsed(), size, epochs));
    }

    println!(
        "{:<12} {:>10} {:>10} {:>12} {:>8}",
        "backend", "prove", "verify", "proof", "epochs"
    );
    for (tag, run) in [("FRI", &fri), ("WHIR", &whir)] {
        if let Some((prove, verify, size, epochs)) = run {
            println!(
                "{tag:<12} {:>9.2}s {:>9.2}s {:>10.2} MiB {epochs:>8}",
                prove.as_secs_f64(),
                verify.as_secs_f64(),
                mib(*size)
            );
        }
    }
    if let (Some(f), Some(w)) = (fri, whir) {
        println!(
            "{:<12} {:>9.2}x {:>9.2}x {:>10.2}x",
            "WHIR/FRI",
            w.0.as_secs_f64() / f.0.as_secs_f64(),
            w.1.as_secs_f64() / f.1.as_secs_f64(),
            w.2 as f64 / f.2 as f64,
        );
    }

    // The pins, after the table. A red one from here costs nothing that was
    // measured.
    #[cfg(feature = "hash-metrics")]
    if let Some((prove_counts, verify_counts, run)) = pinned {
        #[cfg(feature = "cuda")]
        check_device_pins(&bytes, epoch_size_log2, device_commits, &prove_counts);
        check_transcript_pins(
            &bytes,
            epoch_size_log2,
            Some(&run),
            &prove_counts,
            &verify_counts,
        );
    }
}

/// The device counters, pinned the way the transcript pair is: a shape-derived
/// delta on a measured base, behind the same guest guard.
///
/// ★ THE GRIND LINE IS AN IDENTITY, NOT A SECOND NUMBER. The only production
/// readers of `transcript.state()` on this path are `whir_chain`'s `grind` and
/// `check_grind`, one read each, so the device grind count and the prove side's
/// state column are two instruments on one quantity and must agree exactly.
/// Pinning a literal beside the state pin would have been a second copy of the
/// same measurement; this way a host fallback, or a state read appearing
/// somewhere new, says which instrument moved.
///
/// The commit line is not an identity either, but it is not a literal: the
/// bundle's own chains sum to it, one commit per chain round, with DECODE's
/// prepared polynomial held once for the run and its folds paid per epoch, and
/// the genesis stack's held once for the cross-epoch proof with its own folds
/// beside them. See [`transcript_pin::commits`], including the admission
/// assumption it carries.
#[cfg(all(feature = "cuda", feature = "hash-metrics"))]
fn check_device_pins(
    elf: &[u8],
    epoch_size_log2: u32,
    commits: Option<(u64, u64)>,
    prove: &crypto::hash_metrics::Counts,
) {
    use sha2::{Digest, Sha256};

    let sha: String = Sha256::digest(elf)
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect();
    if !pin_applies(
        &sha,
        elf.len(),
        epoch_size_log2,
        crate::tables::max_rows_log2_override(),
    ) {
        println!(
            "{:<12} device pin SKIPPED - see the transcript pin's line",
            "WHIR"
        );
        return;
    }
    let grinds = crypto::grinding::gpu_grind_calls() + crypto::grinding::gpu_grind_calls_rpx();
    assert_eq!(
        grinds, prove.transcript_states,
        "device grinds and the prove side's transcript state reads disagree; on \
         this path every grind reads the state exactly once and nothing else \
         reads it"
    );

    let Some((commits, model)) = commits else {
        panic!("the WHIR arm ran but its device commit count was never taken");
    };
    assert_eq!(
        commits, model,
        "the device commit count moved: the bundle's own chains sum to {model} \
         commits — one per chain round, with DECODE's prepared polynomial held \
         once for the run and its folds paid per epoch, and the genesis stack's \
         held once for the cross-epoch proof — and the counter read {commits}"
    );
    println!(
        "{:<12} device pin OK (commits {commits}, grinds {grinds} = states)",
        "WHIR"
    );
}

/// Where a continuation's time goes: preparing an epoch against proving it.
///
/// Replays what [`crate::multilinear_continuation::prove_continuation`] runs,
/// with a clock on each side of the epoch callback. Preparation is the
/// executor and the trace builders, and it is the half a pipeline can hide
/// behind the previous epoch's proof — which is what the univariate driver
/// does and this one does not. The split is what says whether that is worth
/// building.
#[test]
#[ignore]
fn continuation_phases() {
    use crate::multilinear_continuation;
    use crate::tables::trace_builder::DecodeArtifacts;
    use executor::elf::Elf;

    let name = std::env::var("LAMBDA_VM_BENCH_ELF").unwrap_or_else(|_| "ethrex".into());
    let input = std::env::var("LAMBDA_VM_BENCH_INPUT").unwrap_or_default();
    let epoch_size_log2: u32 = std::env::var("LAMBDA_VM_BENCH_EPOCH_LOG2")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(20);
    let bytes = elf_bytes(&name);
    let inputs = input_bytes(&input);
    let opts = options();
    let label = if input.is_empty() { &name } else { &input };
    println!("\n{label} — continuation phases, epoch_size_log2={epoch_size_log2}");

    let whole = Instant::now();
    let elf = Elf::load(&bytes).expect("load");
    let decode_commitment =
        crate::tables::decode::commitment_from_elf(&elf, &opts).expect("decode commitment");
    let artifacts = DecodeArtifacts::from_elf(&elf).expect("decode artifacts");

    let mut prepare = Vec::new();
    let mut prove = Vec::new();
    let mut epochs = Vec::new();
    let mut last = Instant::now();
    // The dispatch above the epoch loop, as production does it, so DECODE's
    // out-of-band commitment is built once and held for the whole run.
    let boundaries = crate::with_whir_hash!(|H| {
        let pinned = multilinear_continuation::decode_prepared_for::<H>(&elf, &bytes)
            .expect("DECODE's out-of-band commitment");
        crate::continuation::for_each_epoch(
            &elf,
            &inputs,
            epoch_size_log2,
            &artifacts,
            |prepared, _| {
                prepare.push(last.elapsed());
                let start = Instant::now();
                epochs.push(multilinear_continuation::prove_epoch::<H>(
                    &elf,
                    &bytes,
                    &prepared.register_init,
                    prepared.label,
                    prepared.traces,
                    prepared.is_final,
                    &prepared.boundary,
                    &opts,
                    Some(decode_commitment),
                    &pinned,
                )?);
                prove.push(start.elapsed());
                last = Instant::now();
                Ok(())
            },
        )
    })
    .expect("the epochs prepare");

    let start = Instant::now();
    let init_page_data = crate::tables::trace_builder::build_init_page_data(
        &crate::tables::trace_builder::build_initial_image_paged(&elf, &inputs),
    );
    let num_private_input_pages = crate::tables::page::private_input_page_count(&inputs);
    let page_bases = crate::continuation::touched_page_bases(&boundaries);
    // The dispatch is the caller's now, as it already was for the epochs above.
    crate::with_whir_hash!(|H| {
        multilinear_continuation::prove_global::<H>(
            &boundaries,
            &bytes,
            &init_page_data,
            &page_bases,
            num_private_input_pages,
            &opts,
        )
    })
    .expect("the cross-epoch proof");
    let global = start.elapsed();
    let total = whole.elapsed();

    let secs = |d: &std::time::Duration| d.as_secs_f64();
    println!("{:<8} {:>10} {:>10}", "epoch", "prepare", "prove");
    for (i, (p, q)) in prepare.iter().zip(&prove).enumerate() {
        println!("{i:<8} {:>9.2}s {:>9.2}s", secs(p), secs(q));
    }
    let prepared: f64 = prepare.iter().map(secs).sum();
    let proved: f64 = prove.iter().map(secs).sum();
    println!(
        "{:<8} {:>9.2}s {:>9.2}s   cross-epoch {:.2}s   total {:.2}s",
        "sum",
        prepared,
        proved,
        secs(&global),
        secs(&total)
    );
    println!(
        "prepare is {:.0}% of the run — what a pipeline could hide behind the previous proof",
        100.0 * prepared / secs(&total)
    );
    // Which pieces ran on device, summed over every epoch. A count far below
    // the number of tables is a phase above that is a CPU number wearing a GPU
    // label — and over a whole continuation that is easy to miss, because the
    // wall clock grows with the epochs either way.
    #[cfg(feature = "cuda")]
    for (tag, count) in [
        ("gpu commits", multilinear::gpu::commit_calls()),
        ("host fallbacks", multilinear::gpu::host_fallbacks()),
        ("gpu sumchecks", multilinear::gpu::sumcheck_calls()),
        ("gpu evals", multilinear::gpu::evaluate_calls()),
        ("gpu trees", multilinear::gpu::tree_calls()),
        ("gpu factors", multilinear::gpu::factor_calls()),
        ("gpu openings", multilinear::gpu::open_calls()),
    ] {
        println!("{tag:<14} {count:>9}");
    }
}

/// Where the multilinear prover's time goes, phase by phase.
///
/// Replays the same pipeline [`multilinear_prove::prove_with_options_and_inputs`]
/// runs, with a clock between the steps. The split that matters is **commit**
/// (NTT + Merkle, already parallel) against **argue** (sumcheck, GKR, LogUp,
/// stacking — all serial today): the second is the parallelisation backlog, and
/// its share is what says whether closing it is worth the work.
#[test]
#[ignore]
fn phases() {
    use crypto::fiat_shamir::default_transcript::DefaultTranscript;
    use crypto::fiat_shamir::is_transcript::IsTranscript;
    use math::field::element::FieldElement;
    use stark::multilinear_air::Uniforms;
    use stark::multilinear_table::{self, CommittedTable, CommittedTables, TableLayout};

    use crate::test_utils::{E, F};

    let name =
        std::env::var("LAMBDA_VM_BENCH_ELF").unwrap_or_else(|_| "all_instructions_64".into());
    let input = std::env::var("LAMBDA_VM_BENCH_INPUT").unwrap_or_default();
    let bytes = elf_bytes(&name);
    let inputs = input_bytes(&input);
    let opts = options();
    let label = if input.is_empty() { &name } else { &input };
    let threads = std::env::var("RAYON_NUM_THREADS").unwrap_or_else(|_| "all".into());

    let total = Instant::now();
    let start = Instant::now();
    let elf = Elf::load(&bytes).expect("load");
    let logs = Executor::new(&elf, inputs.clone())
        .and_then(Executor::run)
        .expect("run")
        .logs;
    let execute = start.elapsed();

    let start = Instant::now();
    let mut traces = Traces::from_elf_and_logs(
        &elf,
        &logs,
        &MaxRowsConfig::default(),
        &inputs,
        #[cfg(feature = "disk-spill")]
        stark::storage_mode::StorageMode::Ram,
    )
    .expect("traces");
    let trace_build = start.elapsed();

    let table_counts = traces.table_counts();
    let airs = crate::VmAirs::new(
        &elf,
        &opts,
        false,
        &traces.page_configs,
        &table_counts,
        None,
        true,
        None,
        None,
        None,
    );
    let pairs = airs.air_trace_pairs(&mut traces);
    // The table's own shape, the way `multilinear_prove::shapes_of` reads it:
    // transposing the trace to count it is the whole trace copied for two
    // numbers, and it would land outside every phase below.
    let shapes: Vec<(usize, usize)> = pairs
        .iter()
        .map(|(_, trace, _)| {
            (
                trace.main_table.width,
                trace.main_table.height.trailing_zeros() as usize,
            )
        })
        .collect();
    let mut config = multilinear_prove::chain_config(&shapes);
    // `LAMBDA_VM_BENCH_NO_GRIND` zeroes the proof of work while leaving the query
    // count alone. Not a valid proof — it drops the bits grinding buys — but it
    // is the only way to read the grinding cost off the same run, since asking
    // `with_security` for fewer bits would hand them back as extra queries.
    if std::env::var_os("LAMBDA_VM_BENCH_NO_GRIND").is_some() {
        config.grind = GrindBits::default();
    }

    let start = Instant::now();
    let layouts: Vec<TableLayout<'_, F, E>> = pairs
        .iter()
        .zip(&shapes)
        .map(|((air, _, _), &(width, num_vars))| {
            TableLayout::<F, E>::new(
                air.constraint_program(),
                air.constraints_meta(),
                air.bus_interactions(),
                width,
                num_vars,
                Uniforms::default(),
            )
            .expect("layout")
        })
        .collect();
    let layout = start.elapsed();

    // `commit` is now the one commitment the whole proof shares, so this phase
    // is where the stacking shows up.
    let start = Instant::now();
    let tables: Vec<CommittedTable<'_, F, E>> = layouts
        .into_iter()
        .zip(&pairs)
        .map(|(layout, (_, trace, _))| {
            let mut columns = trace.columns_main();
            CommittedTable::from_layout(layout, |col| core::mem::take(&mut columns[col as usize]))
                .expect("materialize")
        })
        .collect();
    let count = tables.len();
    // ⚠ KECCAK ONLY, and it refuses rather than mislabels.
    //
    // The committed tables escape into the rest of this function, so the hash
    // cannot be a `match` here — both arms would have to return the same type
    // and they do not. Rather than print `★ WHIR HASH: rpx256` over keccak's
    // seconds, this bench asserts the knob agrees with what it actually runs.
    // The hash-arm split lives in `commit_phases`, whose `merkle` pass isolates
    // the hashing anyway, which is the number a hash comparison wants.
    assert_eq!(
        crate::whir_hash_knob::selected(),
        crate::whir_hash_knob::Setting::Keccak,
        "`phases` commits with keccak; run `commit_phases` for the hash arms"
    );
    let committed = CommittedTables::<_, _, KeccakWhir>::commit(tables, &config).expect("commit");
    let commit = start.elapsed();

    // `multi_prove`'s own body, so the tables' arguments and the one opening
    // that settles them can be clocked apart: the transcript makes the loop
    // sequential, so this is the same work in the same order.
    let start = Instant::now();
    let mut transcript = DefaultTranscript::<E>::new(&[]);
    for root in committed.roots() {
        transcript.append_bytes(root);
    }
    let z: FieldElement<E> = transcript.sample_field_element();
    let alpha: FieldElement<E> = transcript.sample_field_element();
    let beta: FieldElement<E> = transcript.sample_field_element();
    let mut points: Vec<Vec<FieldElement<E>>> = Vec::new();
    let mut values: Vec<FieldElement<E>> = Vec::new();
    let mut table_proofs = Vec::with_capacity(committed.tables().len());
    for table in committed.tables() {
        let (proof, point) =
            multilinear_table::prove(table, &z, &alpha, &beta, &mut transcript).expect("table");
        for _ in 0..table.num_committed_columns() {
            points.push(point.clone());
        }
        values.extend(proof.constraint.reduce.column_values.iter().cloned());
        table_proofs.push(proof);
    }
    let tables_argued = start.elapsed();

    let start = Instant::now();
    let group_columns: Vec<&multilinear::mle::Mle<F>> = committed
        .tables()
        .iter()
        .flat_map(|t| t.columns())
        .collect();
    let columns = multilinear::stacked_eval::prove::<F, E, _, KeccakWhir>(
        &committed.groups()[0],
        &group_columns,
        None,
        &multilinear::stacked_eval::Claimed::PerColumn(&points),
        &values,
        &config,
        &mut transcript,
    )
    .expect("the opening");
    let opened = start.elapsed();
    let argue = tables_argued + opened;
    let proof = multilinear_table::MultiProof {
        roots: committed.roots().to_vec(),
        tables: table_proofs,
        columns: vec![columns],
        // This bench re-implements `multi_prove`'s body to clock its phases
        // apart; it commits no preprocessed group, so there is nothing to open.
        preprocessed: None,
    };
    let total = total.elapsed();

    println!(
        "\n{label} — CPU, RAYON_NUM_THREADS={threads}, {count} tables, {} queries, {} commitment(s)",
        config.num_queries,
        proof.roots.len(),
    );
    println!("{:<14} {:>9} {:>7}", "phase", "seconds", "share");
    for (tag, took) in [
        ("execute", execute),
        ("trace build", trace_build),
        ("layout", layout),
        ("commit", commit),
        ("argue", argue),
        ("  tables", tables_argued),
        ("  opening", opened),
    ] {
        println!(
            "{tag:<14} {:>9.2} {:>6.1}%",
            took.as_secs_f64(),
            100.0 * took.as_secs_f64() / total.as_secs_f64()
        );
    }
    println!("{:<14} {:>9.2}", "total", total.as_secs_f64());
    // Which pieces of the argument ran on device. A zero here with the `cuda`
    // feature on is the signal that a dispatch declined and the phase above is
    // a CPU number wearing a GPU label.
    #[cfg(feature = "cuda")]
    for (tag, count) in [
        ("gpu grinds", stark::gpu_lde::gpu_grind_calls()),
        ("gpu commits", multilinear::gpu::commit_calls()),
        ("host fallbacks", multilinear::gpu::host_fallbacks()),
        ("gpu sumchecks", multilinear::gpu::sumcheck_calls()),
        ("gpu rounds", multilinear::gpu::sumcheck_rounds()),
        ("gpu evals", multilinear::gpu::evaluate_calls()),
        ("gpu trees", multilinear::gpu::tree_calls()),
        ("gpu factors", multilinear::gpu::factor_calls()),
        ("gpu openings", multilinear::gpu::open_calls()),
    ] {
        println!("{tag:<14} {count:>9}");
    }
    assert_eq!(proof.tables.len(), count);
}

/// Where `commit` goes, pass by pass.
///
/// The phase is one Möbius transform, one NTT and one Merkle commit per stacked
/// polynomial, and only the last two have a kernel already. This says which of
/// the three is worth a kernel first, on the real workload rather than on an
/// operation count.
#[test]
#[ignore]
fn commit_phases() {
    use multilinear::whir::{self, Domain};
    use multilinear::whir_commit::CodewordCommitment;
    use stark::multilinear_air::Uniforms;
    use stark::multilinear_table::{self, TableLayout};

    use crate::test_utils::{E, F};

    let name =
        std::env::var("LAMBDA_VM_BENCH_ELF").unwrap_or_else(|_| "all_instructions_64".into());
    let input = std::env::var("LAMBDA_VM_BENCH_INPUT").unwrap_or_default();
    let bytes = elf_bytes(&name);
    let inputs = input_bytes(&input);
    let elf = Elf::load(&bytes).expect("load");
    let logs = Executor::new(&elf, inputs.clone())
        .and_then(Executor::run)
        .expect("run")
        .logs;
    let mut traces = Traces::from_elf_and_logs(
        &elf,
        &logs,
        &MaxRowsConfig::default(),
        &inputs,
        #[cfg(feature = "disk-spill")]
        stark::storage_mode::StorageMode::Ram,
    )
    .expect("trace");
    let table_counts = traces.table_counts();
    let airs = crate::VmAirs::new(
        &elf,
        &options(),
        false,
        &traces.page_configs,
        &table_counts,
        None,
        true,
        None,
        None,
        None,
    );
    let pairs = airs.air_trace_pairs(&mut traces);

    // The columns that actually get committed are the live ones the layout
    // keeps, not every column of the trace.
    let mut columns: Vec<multilinear::mle::Mle<F>> = Vec::new();
    let mut shapes: Vec<(usize, usize)> = Vec::new();
    for (air, trace, _) in &pairs {
        let main = trace.columns_main();
        let num_vars = main[0].len().trailing_zeros() as usize;
        let layout = TableLayout::<F, E>::new(
            air.constraint_program(),
            air.constraints_meta(),
            air.bus_interactions(),
            main.len(),
            num_vars,
            Uniforms::default(),
        )
        .expect("layout");
        let keys = layout.column_keys().to_vec();
        shapes.push((keys.len(), num_vars));
        for key in keys {
            columns
                .push(multilinear::mle::Mle::new(main[key.col as usize].clone()).expect("column"));
        }
    }
    let config = multilinear_prove::chain_config(&shapes);
    let layout = multilinear_table::global_layout(&shapes).expect("global layout");
    let start = Instant::now();
    let polys = layout
        .stack(&multilinear::stacking::borrow(&columns))
        .expect("stack");
    let stack = start.elapsed();
    drop(columns);

    let mut lift = std::time::Duration::ZERO;
    let mut encode = std::time::Duration::ZERO;
    let mut merkle = std::time::Duration::ZERO;
    for poly in &polys {
        let domain = Domain::<F>::new(poly.num_vars() + config.log_blowup).expect("domain");
        let start = Instant::now();
        let coeffs = whir::lift_coefficients(poly);
        lift += start.elapsed();
        let start = Instant::now();
        let codeword = whir::encode::<F, F>(&coeffs, &domain).expect("encode");
        encode += start.elapsed();
        let start = Instant::now();
        // ★ The `merkle` pass is the LEAF AND TREE HASHING, alone — stack,
        // lift and encode are clocked apart above. So this line is the hash
        // term itself, and it has to follow the knob or the two arms are not
        // comparable.
        crate::with_whir_hash!(|H| {
            let commitment = CodewordCommitment::<_, H>::new(
                &codeword,
                config
                    .schedule(poly.num_vars())
                    .first()
                    .copied()
                    .unwrap_or(0),
            )
            .expect("commit");
            // Dropped inside the arm: the commitment's type names `H`, so it
            // cannot leave the block. That is also why this is the bench the
            // hash arms run through — nothing here escapes.
            drop(commitment);
        });
        merkle += start.elapsed();
    }

    let label = if input.is_empty() { &name } else { &input };
    let total = stack + lift + encode + merkle;
    println!(
        "\n{label} — {} stacked polynomials of 2^{} cells, blowup {}",
        polys.len(),
        layout.n_stack(),
        1 << config.log_blowup,
    );
    println!("{:<14} {:>9} {:>7}", "pass", "seconds", "share");
    for (tag, took) in [
        ("stack", stack),
        ("lift", lift),
        ("encode", encode),
        ("merkle", merkle),
    ] {
        println!(
            "{tag:<14} {:>9.2} {:>6.1}%",
            took.as_secs_f64(),
            100.0 * took.as_secs_f64() / total.as_secs_f64()
        );
    }
    println!("{:<14} {:>9.2}", "total", total.as_secs_f64());
}

/// What a proof is made of, part by part.
///
/// The multilinear proof grows with the **number of tables**, and the suspicion
/// is that the per-table WHIR opening is why: every table commits on its own,
/// so every table pays its own queries. This says how much of the bytes that
/// actually is, which is what decides whether stacking the tables together is
/// worth the work.
#[test]
#[ignore]
fn proof_composition() {
    let name =
        std::env::var("LAMBDA_VM_BENCH_ELF").unwrap_or_else(|_| "all_instructions_64".into());
    let input = std::env::var("LAMBDA_VM_BENCH_INPUT").unwrap_or_default();
    let bytes = elf_bytes(&name);
    let inputs = input_bytes(&input);
    let proof = multilinear_prove::prove_with_options_and_inputs(
        &bytes,
        &inputs,
        &options(),
        &MaxRowsConfig::default(),
    )
    .expect("prove");

    let size = |v: &[u8]| v.len() as f64 / (1024.0 * 1024.0);
    let ser = |x: &dyn Fn() -> Vec<u8>| x();
    let total = rkyv::to_bytes::<rkyv::rancor::Error>(&proof)
        .expect("whole")
        .len();

    // Each part on its own, summed across tables. The opening is not among
    // them: there is one for the whole proof, not one per table.
    let mut gkr = 0usize;
    let mut sumcheck = 0usize;
    let mut reduce = 0usize;
    let mut factor_values = 0usize;
    for t in &proof.proof.tables {
        gkr += rkyv::to_bytes::<rkyv::rancor::Error>(&t.gkr)
            .expect("gkr")
            .len();
        sumcheck += rkyv::to_bytes::<rkyv::rancor::Error>(&t.constraint.sumcheck)
            .expect("sumcheck")
            .len();
        reduce += rkyv::to_bytes::<rkyv::rancor::Error>(&t.constraint.reduce)
            .expect("reduce")
            .len();
        factor_values += rkyv::to_bytes::<rkyv::rancor::Error>(&t.constraint.factor_values)
            .expect("factor_values")
            .len();
    }
    let columns = rkyv::to_bytes::<rkyv::rancor::Error>(&proof.proof.columns)
        .expect("columns")
        .len();
    let _ = ser;

    let label = if input.is_empty() { &name } else { &input };
    println!(
        "\n{label} — {} tables, {} commitment(s)",
        proof.proof.tables.len(),
        proof.proof.roots.len(),
    );

    println!(
        "{:<18} {:>10} {:>8} {:>12}",
        "part", "MiB", "share", "per table"
    );
    for (tag, n) in [
        ("WHIR opening", columns),
        ("GKR", gkr),
        ("sumcheck", sumcheck),
        ("claim reduce", reduce),
        ("factor values", factor_values),
    ] {
        println!(
            "{tag:<18} {:>10.2} {:>7.1}% {:>11.3}",
            size(&vec![0u8; n]),
            100.0 * n as f64 / total as f64,
            size(&vec![0u8; n]) / proof.proof.tables.len() as f64,
        );
    }
    println!("{:<18} {:>10.2}", "whole proof", size(&vec![0u8; total]));
}

/// How big the constraint DAG is per table — the thing `IrShape::combine`
/// walks once per hypercube index.
#[test]
#[ignore]
fn constraint_program_sizes() {
    use crate::test_utils::{E, F};
    let bytes = elf_bytes("ethrex");
    let inputs = input_bytes("ethrex_10_transfers");
    let elf = Elf::load(&bytes).unwrap();
    let logs = Executor::new(&elf, inputs.clone())
        .and_then(Executor::run)
        .unwrap()
        .logs;
    let mut traces = Traces::from_elf_and_logs(
        &elf,
        &logs,
        &MaxRowsConfig::default(),
        &inputs,
        #[cfg(feature = "disk-spill")]
        stark::storage_mode::StorageMode::Ram,
    )
    .unwrap();
    let table_counts = traces.table_counts();
    let airs = crate::VmAirs::new(
        &elf,
        &options(),
        false,
        &traces.page_configs,
        &table_counts,
        None,
        true,
        None,
        None,
        None,
    );
    let pairs = airs.air_trace_pairs(&mut traces);
    // What each table would need resident on a device: its factor tables, which
    // is what the sumcheck folds and the biggest thing a round touches.
    let mut sizes: Vec<(String, usize, usize, usize, f64, f64)> = pairs
        .iter()
        .map(|(air, trace, _)| {
            let columns = trace.columns_main();
            let rows = columns[0].len();
            let layout = stark::multilinear_table::TableLayout::<F, E>::new(
                air.constraint_program(),
                air.constraints_meta(),
                air.bus_interactions(),
                columns.len(),
                rows.trailing_zeros() as usize,
                stark::multilinear_air::Uniforms::default(),
            )
            .expect("layout");
            let factors = layout.kinds().len();
            // ext3 is three Goldilocks limbs.
            let bytes = factors as f64 * rows as f64 * 24.0;
            // The fraction tree: an input layer indexed by (interaction, row),
            // and every level above it — which is one more of the same again.
            let interactions = air.bus_interactions().len().max(1);
            let layer = (rows * interactions.next_power_of_two()) as f64;
            let tree = layer * 2.0 * 2.0 * 24.0;
            (
                air.name().to_string(),
                air.constraint_program().nodes.len(),
                rows,
                factors,
                bytes / (1u64 << 30) as f64,
                tree / (1u64 << 30) as f64,
            )
        })
        .collect();
    let factors_total: f64 = sizes.iter().map(|s| s.4).sum();
    let tree_max = sizes.iter().map(|s| s.5).fold(0.0f64, f64::max);
    sizes.sort_by(|a, b| (b.4 + b.5).partial_cmp(&(a.4 + a.5)).unwrap());
    sizes.dedup_by(|a, b| a.0 == b.0);
    println!(
        "\n{:<22} {:>10} {:>9} {:>8} {:>10} {:>10}",
        "table", "DAG nodes", "rows", "factors", "ext3 GiB", "GKR GiB"
    );
    for (name, nodes, rows, factors, gib, tree) in sizes.iter().take(10) {
        println!("{name:<22} {nodes:>10} {rows:>9} {factors:>8} {gib:>9.2} {tree:>9.2}");
    }
    println!("\nfactores de todas las tablas juntos: {factors_total:.2} GiB");
    println!("arbol de fracciones mas grande:      {tree_max:.2} GiB");
}

/// ★ M2 (lane V1, uncommitted measurement): the exact WHIR stack shape of a real
/// epoch — per-table `(name, width, rows)`, then `n_stack` / `num_polys` per
/// commitment group, the rounds each chain runs and the query count the config
/// derives. Pure shape logic on top of the executor; no card, no proof.
#[test]
#[ignore]
fn whir_epoch_shapes() {
    use crate::multilinear_continuation::{epoch_groups, global_groups};
    use crate::multilinear_prove::{chain_config, stacks};
    use crate::tables::trace_builder::DecodeArtifacts;
    use executor::elf::Elf;

    let name = std::env::var("LAMBDA_VM_BENCH_ELF").unwrap_or_else(|_| "ethrex".into());
    let input = std::env::var("LAMBDA_VM_BENCH_INPUT").unwrap_or_default();
    let epoch_size_log2: u32 = std::env::var("LAMBDA_VM_BENCH_EPOCH_LOG2")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(21);
    let bytes = elf_bytes(&name);
    let inputs = input_bytes(&input);
    let opts = options();
    let elf = Elf::load(&bytes).expect("load");
    let artifacts = DecodeArtifacts::from_elf(&elf).expect("decode artifacts");
    println!("\n== M2 shapes: {input} epoch_size_log2={epoch_size_log2} ==");

    let mut totals: Vec<(u64, usize, usize, usize, usize, usize)> = Vec::new();
    let boundaries = crate::continuation::for_each_epoch(
        &elf,
        &inputs,
        epoch_size_log2,
        &artifacts,
        |prepared, _| {
            let mut traces = prepared.traces;
            crate::tables::bitwise::update_multiplicities(
                &mut traces.bitwise,
                &crate::tables::local_to_global::collect_bitwise_from_l2g(&prepared.boundary),
            );
            let reg_fini = crate::tables::register::fini_from_trace(&traces.register);
            let table_counts = traces.table_counts();
            let airs = crate::continuation::build_epoch_airs(
                &elf,
                &opts,
                &[],
                &table_counts,
                &prepared.register_init,
                &reg_fini,
                prepared.is_final,
                None,
            );
            let l2g_air = crate::continuation::l2g_memory_air(&opts, prepared.label);
            let mut l2g_trace =
                crate::tables::local_to_global::generate_local_to_global_trace(&prepared.boundary);
            let mut pairs = airs.air_trace_pairs(&mut traces);
            pairs.push((&l2g_air, &mut l2g_trace, &()));

            let shapes: Vec<(usize, usize)> = pairs
                .iter()
                .map(|(_, t, _)| {
                    (
                        t.main_table.width,
                        t.main_table.height.trailing_zeros() as usize,
                    )
                })
                .collect();
            if prepared.index == 0 {
                println!("\n-- epoch 0 per-table census --");
                println!("{:<16} {:>7} {:>10} {:>6} {:>14}", "table", "width", "rows", "vars", "cells");
                for ((air, _, _), &(w, v)) in pairs.iter().zip(&shapes) {
                    println!(
                        "{:<16} {w:>7} {:>10} {v:>6} {:>14}",
                        air.name(),
                        1usize << v,
                        (w as u64) << v
                    );
                }
            }
            let config = chain_config(&shapes);
            let sizes = epoch_groups(shapes.len());
            let (layouts, _d) = stacks(&shapes, &sizes, &config).expect("stacks");
            let cells: u64 = shapes.iter().map(|&(w, v)| (w as u64) << v).sum();
            let chains: usize = layouts.iter().map(|l| l.num_polys()).sum();
            let rounds: usize = layouts
                .iter()
                .map(|l| l.num_polys() * l.n_stack().div_ceil(config.log_folding))
                .sum();
            println!(
                "epoch {:>2}: tables {:>2} cells {:>12} | group0 n_stack {:>2} polys {:>2} | bookend n_stack {:>2} polys {:>2} | chains {:>2} rounds {:>3} | Q {}",
                prepared.index,
                shapes.len(),
                cells,
                layouts[0].n_stack(),
                layouts[0].num_polys(),
                layouts[1].n_stack(),
                layouts[1].num_polys(),
                chains,
                rounds,
                config.num_queries,
            );
            totals.push((
                prepared.index,
                shapes.len(),
                cells as usize,
                chains,
                rounds,
                config.num_queries,
            ));
            Ok(())
        },
    )
    .expect("epochs prepare");

    // The cross-epoch proof's groups: one per bookend, then the global-memory tables.
    let init_page_data = crate::tables::trace_builder::build_init_page_data(
        &crate::tables::trace_builder::build_initial_image_paged(&elf, &inputs),
    );
    let num_private_input_pages = crate::tables::page::private_input_page_count(&inputs);
    let page_bases = crate::continuation::touched_page_bases(&boundaries);
    let gm_configs = crate::continuation::global_memory_configs_from_init_page_data(
        &page_bases,
        &init_page_data,
        num_private_input_pages,
        true,
    );
    let l2g_airs: Vec<_> = (0..boundaries.len())
        .map(|i| {
            crate::continuation::l2g_global_air(
                &opts,
                crate::tables::local_to_global::epoch_label(i as u64),
            )
        })
        .collect();
    let gm_airs: Vec<_> = gm_configs
        .iter()
        .map(|c| crate::continuation::global_memory_air(&opts, c, None))
        .collect();
    let mut l2g_traces: Vec<_> = boundaries
        .iter()
        .map(|e| crate::tables::local_to_global::generate_local_to_global_trace(e.as_slice()))
        .collect();
    let mut final_state: crate::tables::global_memory::FiniStateMap =
        std::collections::HashMap::new();
    for epoch in &boundaries {
        for b in epoch.iter() {
            final_state.insert(
                b.address,
                crate::tables::global_memory::FiniState {
                    value: (b.fini.value & 0xFF) as u8,
                    epoch: b.fini.epoch,
                },
            );
        }
    }
    let mut gm_traces: Vec<_> = gm_configs
        .iter()
        .map(|c| crate::tables::global_memory::generate_global_trace(c, &final_state))
        .collect();
    let mut gpairs: Vec<crate::AirTracePair<'_>> = Vec::new();
    for (a, t) in l2g_airs.iter().zip(l2g_traces.iter_mut()) {
        gpairs.push((a, t, &()));
    }
    for (a, t) in gm_airs.iter().zip(gm_traces.iter_mut()) {
        gpairs.push((a, t, &()));
    }
    let gshapes: Vec<(usize, usize)> = gpairs
        .iter()
        .map(|(_, t, _)| {
            (
                t.main_table.width,
                t.main_table.height.trailing_zeros() as usize,
            )
        })
        .collect();
    let gconfig = chain_config(&gshapes);
    let gsizes = global_groups(boundaries.len(), gm_configs.len());
    let (glayouts, _gd) = stacks(&gshapes, &gsizes, &gconfig).expect("global stacks");
    let gchains: usize = glayouts.iter().map(|l| l.num_polys()).sum();
    let grounds: usize = glayouts
        .iter()
        .map(|l| l.num_polys() * l.n_stack().div_ceil(gconfig.log_folding))
        .sum();
    println!(
        "\nGLOBAL: pages {} tables {} groups {} chains {} rounds {} Q {}",
        gm_configs.len(),
        gshapes.len(),
        gsizes.len(),
        gchains,
        grounds,
        gconfig.num_queries
    );
    for (i, l) in glayouts.iter().enumerate().take(3) {
        println!(
            "   group {i}: n_stack {} polys {}",
            l.n_stack(),
            l.num_polys()
        );
    }
    if let Some(l) = glayouts.last() {
        println!(
            "   group {} (global memory): n_stack {} polys {}",
            glayouts.len() - 1,
            l.n_stack(),
            l.num_polys()
        );
    }
    let ec: usize = totals.iter().map(|t| t.3).sum();
    let er: usize = totals.iter().map(|t| t.4).sum();
    println!(
        "\n★ TOTAL over {} epochs + global: chains {} rounds {} | grinds (3R-1 per chain) {}",
        totals.len(),
        ec + gchains,
        er + grounds,
        3 * (er + grounds) - (ec + gchains)
    );
}

/// ★ The per-table AIR properties V1's census needs, which `whir_epoch_shapes`
/// does not print: the interaction count, the factor layout, the frame offsets
/// and the constraint degree.
///
/// All four are EXECUTION-INDEPENDENT — they are properties of the AIR and of
/// the table's width, not of what the guest did — but building the AIR set
/// needs the guest ELF, so this is a box run and not a laptop one. It is a
/// sibling of `whir_epoch_shapes` rather than an edit of it, so that
/// instrument's log stays the one sh1 filed.
///
/// What each column is for:
///
/// - `I` sets the GKR ladder's height: `input_layer_vars(I, num_vars)` rounds,
///   and the bus weights are `eq_evals` over `ceil(log2 I)` variables, which is
///   the one term of a table's cost that is exponential in anything;
/// - `factors` is what `claim_reduce` batches and `offsets` is how many `shift`
///   kernels it spends — one per DISTINCT offset, however many factors read it;
/// - `degree` is the zerocheck rule's, so the main batched sumcheck runs at
///   `max(degree + 1, 2)` (`multilinear_table.rs:765`, `batch.rs:264`);
/// - `roots` is the length `beta_powers` must have.
#[test]
#[ignore = "needs the guest ELF and builds every epoch's AIRs"]
fn whir_table_shapes() {
    use crate::tables::trace_builder::DecodeArtifacts;
    use executor::elf::Elf;
    use multilinear::constraint_argument::FactorKind;
    use stark::multilinear_air::Uniforms;
    use stark::multilinear_table::TableLayout;

    let name = std::env::var("LAMBDA_VM_BENCH_ELF").unwrap_or_else(|_| "ethrex".into());
    let input = std::env::var("LAMBDA_VM_BENCH_INPUT").unwrap_or_default();
    let epoch_size_log2: u32 = std::env::var("LAMBDA_VM_BENCH_EPOCH_LOG2")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(21);
    let bytes = elf_bytes(&name);
    let inputs = input_bytes(&input);
    let opts = options();
    let elf = Elf::load(&bytes).expect("load");
    let artifacts = DecodeArtifacts::from_elf(&elf).expect("decode artifacts");
    println!("\n== V1 table shapes: {input} epoch_size_log2={epoch_size_log2} ==");

    crate::continuation::for_each_epoch(
        &elf,
        &inputs,
        epoch_size_log2,
        &artifacts,
        |prepared, _| {
            if prepared.index != 0 {
                return Ok(());
            }
            let mut traces = prepared.traces;
            crate::tables::bitwise::update_multiplicities(
                &mut traces.bitwise,
                &crate::tables::local_to_global::collect_bitwise_from_l2g(&prepared.boundary),
            );
            let reg_fini = crate::tables::register::fini_from_trace(&traces.register);
            let table_counts = traces.table_counts();
            let airs = crate::continuation::build_epoch_airs(
                &elf,
                &opts,
                &[],
                &table_counts,
                &prepared.register_init,
                &reg_fini,
                prepared.is_final,
                None,
            );
            let l2g_air = crate::continuation::l2g_memory_air(&opts, prepared.label);
            let mut l2g_trace =
                crate::tables::local_to_global::generate_local_to_global_trace(&prepared.boundary);
            let mut pairs = airs.air_trace_pairs(&mut traces);
            pairs.push((&l2g_air, &mut l2g_trace, &()));

            println!(
                "{:<16} {:>5} {:>6} {:>4} {:>8} {:>8} {:>7} {:>7} {:>7} offsets",
                "table", "vars", "width", "I", "columns", "factors", "public", "degree", "roots"
            );
            let mut degree_rounds = 0usize;
            for (air, trace, _) in pairs.iter() {
                let width = trace.main_table.width;
                let num_vars = trace.main_table.height.trailing_zeros() as usize;
                let layout = TableLayout::<
                    crate::tables::types::GoldilocksField,
                    crate::tables::types::GoldilocksExtension,
                >::new(
                    air.constraint_program(),
                    air.constraints_meta(),
                    air.bus_interactions(),
                    width,
                    num_vars,
                    Uniforms::default(),
                )
                .expect("the table lays out");
                let kinds = layout.kinds();
                let public = kinds
                    .iter()
                    .filter(|k| matches!(k, FactorKind::Public))
                    .count();
                let mut offsets: Vec<usize> = kinds
                    .iter()
                    .filter_map(|k| k.source().map(|s| s.offset))
                    .collect();
                offsets.sort_unstable();
                offsets.dedup();
                let degree = layout.shape().degree();
                degree_rounds += num_vars;
                println!(
                    "{:<16} {num_vars:>5} {width:>6} {:>4} {:>8} {:>8} {public:>7} {degree:>7} {:>7} {offsets:?}",
                    air.name(),
                    air.bus_interactions().len(),
                    layout.num_columns(),
                    kinds.len(),
                    layout.shape().num_roots(),
                );
            }
            println!("epoch 0: {} tables, {degree_rounds} main-sumcheck rounds over all of them", pairs.len());
            Ok(())
        },
    )
    .expect("epochs prepare");
}

/// ★ THE PER-TABLE HALF of item 5's recount, in one card-free run.
///
/// `whir_table_shapes` gives the interaction COUNT; what the bus statements
/// actually cost is set by what is INSIDE those interactions — how many bus
/// elements each carries and how many columns each element reads — and neither
/// is derivable from `I`. This prints those, and then the whole thing:
/// `whir_table::table_verify_cost` for every epoch-0 table, rows and
/// permutations, with the SPONGE THREADED across the tables the way an
/// assembled epoch threads it rather than restarted fresh at each one.
///
/// ⚠ What the total is NOT: it is the per-table half alone. The chain half, the
/// DECODE group's own chain, `stacked_eval`'s wrapper, the epoch statement and
/// the preprocessed seam are each named in the output rather than estimated
/// into it. ⚠ And these rows INCLUDE their constants, where `chain_rows` does
/// not — the two conventions are set out in V1e's handoff and must not be added
/// without saying which one the sum is in.
///
/// Execution-independent (the AIRs' own structure), card-free, and it returns
/// after epoch 0 — but building the AIR set needs the guest ELF, so it is a box
/// run. A SIBLING of `whir_table_shapes` rather than an edit of it, so that
/// instrument's filed log stays the one ts1 produced.
///
/// ```text
/// LAMBDA_VM_BENCH_ELF=ethrex LAMBDA_VM_BENCH_INPUT=ethrex_mainnet_25368371 \
/// LAMBDA_VM_BENCH_EPOCH_LOG2=21 \
/// cargo test --release -p lambda-vm-prover --lib whir_bus_shapes -- --ignored --nocapture
/// ```
#[test]
#[ignore = "needs the guest ELF and builds every epoch's AIRs"]
fn whir_bus_shapes() {
    use crate::lfm::whir_bus::claim_statements_cost;
    use crate::lfm::whir_table::{TableShape, table_verify_cost};
    use crate::lfm::whir_transcript::SpongeEntry;
    use crate::tables::trace_builder::DecodeArtifacts;
    use executor::elf::Elf;
    use multilinear::Error as MlError;
    use stark::multilinear_air::Uniforms;
    use stark::multilinear_logup::interaction_shapes;
    use stark::multilinear_table::TableLayout;

    let name = std::env::var("LAMBDA_VM_BENCH_ELF").unwrap_or_else(|_| "ethrex".into());
    let input = std::env::var("LAMBDA_VM_BENCH_INPUT").unwrap_or_default();
    let epoch_size_log2: u32 = std::env::var("LAMBDA_VM_BENCH_EPOCH_LOG2")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(21);
    let bytes = elf_bytes(&name);
    let inputs = input_bytes(&input);
    let opts = options();
    let elf = Elf::load(&bytes).expect("load");
    let artifacts = DecodeArtifacts::from_elf(&elf).expect("decode artifacts");
    println!("\n== V1 bus shapes: {input} epoch_size_log2={epoch_size_log2} ==");

    crate::continuation::for_each_epoch(
        &elf,
        &inputs,
        epoch_size_log2,
        &artifacts,
        |prepared, _| {
            if prepared.index != 0 {
                return Ok(());
            }
            let mut traces = prepared.traces;
            crate::tables::bitwise::update_multiplicities(
                &mut traces.bitwise,
                &crate::tables::local_to_global::collect_bitwise_from_l2g(&prepared.boundary),
            );
            let reg_fini = crate::tables::register::fini_from_trace(&traces.register);
            let table_counts = traces.table_counts();
            let airs = crate::continuation::build_epoch_airs(
                &elf,
                &opts,
                &[],
                &table_counts,
                &prepared.register_init,
                &reg_fini,
                prepared.is_final,
                None,
            );
            let l2g_air = crate::continuation::l2g_memory_air(&opts, prepared.label);
            let mut l2g_trace =
                crate::tables::local_to_global::generate_local_to_global_trace(&prepared.boundary);
            let mut pairs = airs.air_trace_pairs(&mut traces);
            pairs.push((&l2g_air, &mut l2g_trace, &()));

            println!(
                "{:<16} {:>5} {:>5} {:>9} {:>6} {:>7} {:>8} {:>5} {:>4} {:>10} {:>8} {:>6}",
                "table",
                "vars",
                "I",
                "elements",
                "terms",
                "widest",
                "bus rows",
                "gkr",
                "deg",
                "table rows",
                "perms",
                "pool"
            );
            let (mut ti, mut te, mut tt, mut tw, mut tr) = (0usize, 0usize, 0usize, 0usize, 0usize);
            let (mut total_rows, mut total_perms) = (0usize, 0usize);
            // ★ THE COLUMN THE RECOUNT NEEDED AND bs2 DID NOT PRINT. Each
            // table's `pool` is how many of its rows are `LFM_CONST`, so the
            // CONST-FREE per-table half is `Σ rows − Σ pool`; and `one pool` is
            // the UNION of those words, which is what an assembled epoch
            // actually pays, because one program has one constant pool and the
            // tables' constants collide in it. Printing both is what closes the
            // convention gap: the recount's 2,064,462 is neither figure, and
            // with these two lines it can be stated in either.
            let mut union: Vec<crate::lfm::word::LfmWord> = Vec::new();
            let mut pool_sum = 0usize;
            // The sponge is THREADED, as an assembled epoch threads it: a table
            // enters on whatever the one before it left, not on `fresh`.
            let mut entry = SpongeEntry::fresh();
            for (air, trace, _) in pairs.iter() {
                let width = trace.main_table.width;
                let num_vars = trace.main_table.height.trailing_zeros() as usize;
                let layout = TableLayout::<
                    crate::tables::types::GoldilocksField,
                    crate::tables::types::GoldilocksExtension,
                >::new(
                    air.constraint_program(),
                    air.constraints_meta(),
                    air.bus_interactions(),
                    width,
                    num_vars,
                    Uniforms::default(),
                )
                .expect("the table lays out");
                let slots = layout.slot_of().to_vec();
                let shapes = interaction_shapes::<crate::tables::types::GoldilocksExtension, _>(
                    air.bus_interactions(),
                    width,
                    |column| {
                        slots
                            .get(column)
                            .copied()
                            .ok_or(MlError::UnknownPolynomial {
                                index: column,
                                len: slots.len(),
                            })
                    },
                )
                .expect("the bus probes");

                let elements: usize = shapes.iter().map(|s| s.elements.len()).sum();
                let terms: usize = shapes
                    .iter()
                    .map(|s| {
                        s.numerator.terms().len()
                            + s.elements.iter().map(|e| e.terms().len()).sum::<usize>()
                    })
                    .sum();
                let widest = shapes.iter().map(|s| s.elements.len()).max().unwrap_or(0);
                let cost = claim_statements_cost(&shapes, num_vars);
                let shape = TableShape {
                    ir: layout.shape(),
                    bus: &shapes,
                    kinds: layout.kinds(),
                    num_columns: layout.num_columns(),
                    num_vars,
                };
                let whole = table_verify_cost(&shape, entry);
                entry = whole.entry();
                println!(
                    "{:<16} {num_vars:>5} {:>5} {elements:>9} {terms:>6} {widest:>7} {:>8} {:>5} {:>4} {:>10} {:>8} {:>6}",
                    air.name(),
                    shapes.len(),
                    cost.rows(),
                    shape.gkr_layers(),
                    shape.sumcheck_degree(),
                    whole.rows(),
                    whole.perms(),
                    whole.leg.constant_values().len(),
                );
                ti += shapes.len();
                te += elements;
                tt += terms;
                tw = tw.max(widest);
                tr += cost.rows();
                total_rows += whole.rows();
                total_perms += whole.perms();
                pool_sum += whole.leg.constant_values().len();
                for word in whole.leg.constant_values() {
                    if !union.contains(word) {
                        union.push(*word);
                    }
                }
            }
            println!(
                "epoch 0 BUS TOTAL over {} tables: I {ti} | elements {te} | affine terms {tt} \
                 | widest interaction {tw} | claim_statements rows {tr}",
                pairs.len()
            );
            println!(
                "★ epoch 0 PER-TABLE HALF over {} tables: {total_rows} rows | {total_perms} \
                 permutations",
                pairs.len()
            );
            println!(
                "★ epoch 0 CONSTANTS: {pool_sum} summed over the tables | {} in the ONE pool an \
                 assembled epoch has | CONST-FREE per-table half {} rows | ONE-POOL per-table \
                 half {} rows",
                union.len(),
                total_rows - pool_sum,
                total_rows - pool_sum + union.len(),
            );
            println!(
                "  ⚠ NOT the whole epoch. Missing, and each is named rather than estimated: the \
                 CHAIN half (V1d: 1,484,072 rows over eight chains at n_stack 25, a CONST-FREE \
                 figure — see V1e's handoff on the two conventions), the DECODE group's own chain \
                 per epoch, `stacked_eval`'s wrapper (item 3), the epoch STATEMENT (item 4), and \
                 the preprocessed seam (BITWISE's landed 295-instruction closed form, DECODE's \
                 pinned group). The rows above INCLUDE their constants."
            );
            Ok(())
        },
    )
    .expect("epochs prepare");
}
