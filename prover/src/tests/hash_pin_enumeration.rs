//! ★ **THE ENUMERATION GATE** — the hash must be NAMED, never implied.
//!
//! `lfm/SOUNDNESS.md` §6.7 states the rule: every prove, verify, execute and
//! commitment-building call on the block path names [`crate::hash_pin`] rather
//! than a workspace default alias. Eleven sites were carrying an implied hash
//! and were closed by enumerating call sites BY HAND — and a hand enumeration
//! is worth exactly as long as nobody adds a twelfth.
//!
//! ⚠ **This is a blessed-set test, not a ban.** Three of the four symbols have
//! legitimate uses that are not on the block path: test helpers that build a
//! trace or a fixture under the toy permutation, host-side byte-transcript
//! differentials, and one assertion whose whole content is what the default IS.
//! Each blessed file carries its reason below. A NEW file appearing fails the
//! test and names itself, which forces the reachability question — *is this on
//! the block path?* — to be answered by a person, once, rather than assumed.
//!
//! ⚖ ASSESSMENT of what it does NOT catch: a call site can name a hash
//! explicitly and name the WRONG one, and a source scan cannot see that. The
//! instruments for that are the pin's own coherence tests
//! (`hash_pin::tests`) and the differentials in `algebraic_commit` /
//! `algebraic_transcript`. This gate closes the *silent default*, which is the
//! failure mode that produced all eleven.

use std::collections::BTreeSet;
use std::path::Path;

/// The symbols that silently select a hash when nobody names one.
const IMPLIED_HASH_SYMBOLS: &[&str] = &[
    // The workspace's commitment configuration, and the `Prover` / `Verifier`
    // aliases that are `GenericProver` / `GenericVerifier` AT it.
    "DefaultStarkHash",
    // The workspace's Fiat-Shamir transcript OBJECT — §6.7's axis 2, the one
    // whose half-flip is SILENT.
    "DefaultStarkTranscript",
    // The `LFM_HASH` socket permutation — §6.7's axis 3. The default is `Test`,
    // a one-round toy.
    "HasherKind::default()",
];

/// Files allowed to mention an implied-hash symbol, each with its reason.
///
/// Paths are relative to `prover/src`. Two files are excluded from the scan
/// rather than blessed: `hash_pin.rs`, because naming the default is what it is
/// FOR, and this file, because it has to spell the symbols it searches for.
///
/// ⚠ Scope is `prover/src` only. `crypto/stark` DEFINES the aliases, so
/// scanning it would return the definitions and every doc line about them; the
/// pin is a `prover`-crate concept and the call sites that matter are here.
const BLESSED: &[(&str, &str)] = &[
    (
        "lfm/airs.rs",
        "`lfm_chip_census` / `lfm_cell_counts` / `LfmAirs::new` default the \
         socket hasher. ✓ VERIFIED test-only: the census pair counts cells and \
         proves nothing, and `LfmAirs::new` has exactly one caller \
         (`wrap_tests.rs`). Production builds its AIR set through \
         `LfmAirs::new_with_hasher`.",
    ),
    (
        "lfm/trace.rs",
        "`build_traces` defaults the socket hasher. ✓ VERIFIED test-only \
         (`wrap_tests`, `blake3_chip_tests`, `machine_tests`); production \
         reaches `build_traces_with_hasher` through \
         `proof::lfm_prove_with_hasher`, which passes ONE hasher to the \
         executor, the AIR set and the trace builder.",
    ),
    (
        "lfm/fixture.rs",
        "Fixture construction for the chip suites, under the toy permutation \
         by design.",
    ),
    (
        "lfm/rpo_chip_tests.rs",
        "Asserts `HasherKind::default() == HasherKind::Test` — the assertion's \
         whole content is what the default is.",
    ),
    ("lfm/poseidon_chip_tests.rs", "As `rpo_chip_tests.rs`."),
    (
        "lfm/machine_tests.rs",
        "Host-side BYTE-transcript differentials: the oracle for the machine's \
         byte `TranscriptReplay` arm is deliberately the byte transcript.",
    ),
    (
        "tests/prove_elfs_tests.rs",
        "Names `DefaultStarkTranscript` deliberately — its header records that \
         the production path's transcript must be the one the default \
         commitment configuration names, and the test exists to hold that.",
    ),
    (
        "tests/recursion_soundness_gap_poc.rs",
        "A proof-of-concept against the workspace default configuration.",
    ),
    (
        "tests/page_offset_forgery_poc.rs",
        "As `recursion_soundness_gap_poc.rs`.",
    ),
];

/// Every `.rs` under `dir`, relative to `root`.
fn rust_files(root: &Path, dir: &Path, out: &mut Vec<std::path::PathBuf>) {
    for entry in std::fs::read_dir(dir).expect("the crate source must be readable") {
        let path = entry.expect("a readable dir entry").path();
        if path.is_dir() {
            rust_files(root, &path, out);
        } else if path.extension().is_some_and(|e| e == "rs") {
            out.push(
                path.strip_prefix(root)
                    .expect("a path under the root")
                    .to_path_buf(),
            );
        }
    }
}

/// A line with its trailing `//` comment removed, or `None` if the whole line
/// is a comment.
///
/// ⚠ Deliberately crude — a `//` inside a string literal would truncate the
/// line early. That direction is safe: it can only make the scan miss a
/// mention, and every mention this gate is about is real code. What it must
/// NOT do is count prose, because the modules that explain this rule discuss
/// the symbols by name in nearly every paragraph.
fn code_of(line: &str) -> Option<&str> {
    let t = line.trim_start();
    if t.starts_with("//") || t.starts_with('*') {
        return None;
    }
    Some(match line.find("//") {
        Some(i) => &line[..i],
        None => line,
    })
}

#[test]
fn no_call_site_outside_the_pin_reaches_a_default_alias() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("src");
    let mut files = Vec::new();
    rust_files(&root, &root, &mut files);
    assert!(
        files.len() > 50,
        "the scan must actually see the crate, found {} files",
        files.len()
    );

    let mut found: BTreeSet<String> = BTreeSet::new();
    for rel in &files {
        // `hash_pin.rs` names the defaults because naming them is what it is
        // FOR; this file names them because it has to spell what it searches
        // for. Both are excluded by identity rather than by a pattern, so a
        // third file cannot join them by accident.
        if rel == Path::new("hash_pin.rs") || rel == Path::new("tests/hash_pin_enumeration.rs") {
            continue;
        }
        let text = std::fs::read_to_string(root.join(rel)).expect("a readable source file");
        for line in text.lines() {
            let Some(code) = code_of(line) else { continue };
            if IMPLIED_HASH_SYMBOLS.iter().any(|s| code.contains(s)) {
                found.insert(rel.to_string_lossy().replace('\\', "/"));
            }
        }
    }

    let blessed: BTreeSet<String> = BLESSED.iter().map(|(f, _)| (*f).to_string()).collect();
    let unexpected: Vec<&String> = found.difference(&blessed).collect();
    assert!(
        unexpected.is_empty(),
        "★ a NEW site reaches an implied hash: {unexpected:?}\n\
         Every prove / verify / execute / commitment-building call on the block \
         path must name `crate::hash_pin` (SOUNDNESS.md §6.7). If this site is \
         genuinely off the block path, add it to BLESSED with the reason it is \
         — and check the reachability, because the last eleven all looked off \
         the path too."
    );
    let stale: Vec<&String> = blessed.difference(&found).collect();
    assert!(
        stale.is_empty(),
        "a blessed file no longer mentions an implied hash — drop it from \
         BLESSED so the list stays an inventory rather than a wish: {stale:?}"
    );
}
