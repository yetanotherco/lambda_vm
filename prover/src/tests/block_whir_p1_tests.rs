//! D-WHIR-P1 S3, D-P3B P3: the block's base is its proof family's
//! ([`crate::hash_pin::BlockFamily`]): [`BaseFormat::P1_WHIR`], ZisK's
//! Poseidon1 (4-ary trees capped at height 2, its field sponge and W8 grind).
//! The format names it, the family pins it: any other base — RPX's included —
//! is refused; a P1 proof verifies under no other base; and the statement's tag
//! names the base's geometry.

use crate::block_whir::{
    self, BlockFormat, BlockOptions, BlockWhirProof, Deviations, MAX_P1_CAP_HEIGHT, P1_TAG_OMITTED,
    block_statement_tag, checked_base, prove_block_whir_with, verify_block_whir,
};
use crate::statement::MULTILINEAR_BLOCK_TAG;
use crate::tables::MaxRowsConfig;
use crate::test_utils::asm_elf_bytes;
use crate::zf_format::ZfFormat;
use multilinear::whir_chain::{ArgueFormat, StackVars};
use stark::config::CommitmentHash;
use stark::proof::options::{BaseFormat, CapPolicy, ProofOptions};

/// A stack of 2^10 and two polynomials a group, so a small program spans
/// several groups, under `base`.
fn many_groups(base: BaseFormat) -> BlockFormat {
    let mut zf = ZfFormat::DEFAULT.with_base(base);
    zf.whir_stack = StackVars::new(10).expect("a valid stack");
    BlockFormat {
        zf,
        group_polys: 2,
        max_groups: block_whir::BLOCK_MAX_GROUPS,
        prepared: true,
        argue: ArgueFormat::PerTable,
        group: multilinear::whir_chain::GroupFold::PerPoly,
    }
}

/// `block_whir_tests`' small-program options, `drop_levels` dropped from each
/// retired tree (binary levels; half as many 4-ary ones under P1).
fn options(drop_levels: usize) -> BlockOptions {
    BlockOptions {
        max_rows: MaxRowsConfig::small(),
        keccak_rnd_rows_log2: 16,
        ecdas_rows_log2: block_whir::BLOCK_ECDAS_ROWS_LOG2,
        keccak_rows_log2: block_whir::BLOCK_KECCAK_ROWS_LOG2,
        ecsm_rows_log2: block_whir::BLOCK_ECSM_ROWS_LOG2,
        copy_rows_log2: block_whir::BLOCK_COPY_ROWS_LOG2,
        drop_levels: multilinear::whir_commit::TreeDrop::uniform(drop_levels),
        window_log2: None,
        stream_keccak_rnd: false,
        stream_memw_lt: false,
        drop_streamed_ops: false,
        layout_workers: 0,
        layout_ahead: Some(2),
        pack_rest_as_laid_out: false,
        narrow: stark::multilinear_block::Narrowing::Card { min_cells: 0 },
        upload_ahead: true,
        memlog: false,
        finish_keccak_rnd_chunks: true,
        finish_cuts: true,
        finish_stream: true,
        rest_layout_bytes: Some(1 << 20),
        pack_finished: true,
        gpack: true,
        spill: block_whir::BlockSpillPolicy::Off,
        regen: None,
    }
}

fn prove(elf: &[u8], format: &BlockFormat, drop_levels: usize) -> BlockWhirProof {
    prove_block_whir_with(
        elf,
        &[],
        &ProofOptions::default_test_options(),
        format,
        &options(drop_levels),
        &Deviations::default(),
    )
    .expect("prove")
    .0
}

fn verify(proof: &BlockWhirProof, elf: &[u8], format: &BlockFormat) -> bool {
    verify_block_whir(proof, elf, &ProofOptions::default_test_options(), format).unwrap_or(false)
}

/// The tag names the base's effective geometry under the family's hash: no
/// base, RPX's included, gets the untagged V1 tag.
#[test]
fn the_statement_tag_names_the_bases_geometry() {
    let p1 = |cap| BaseFormat {
        hash: CommitmentHash::Poseidon1,
        arity4_cap: cap,
    };
    let tag = |cap| String::from_utf8(block_statement_tag(&p1(cap))).expect("ascii");
    let v1 = String::from_utf8(MULTILINEAR_BLOCK_TAG.to_vec()).expect("ascii");
    assert_ne!(block_statement_tag(&BaseFormat::RPX), MULTILINEAR_BLOCK_TAG);
    assert_eq!(tag(CapPolicy::Fixed(2)), format!("{v1}/P1W16/C2"));
    assert_eq!(
        block_statement_tag(&BaseFormat::P1_WHIR),
        tag(CapPolicy::Fixed(2)).as_bytes()
    );
    assert_eq!(tag(CapPolicy::Off), format!("{v1}/P1W16/C0"));
    assert_eq!(
        tag(CapPolicy::Fixed(0)),
        tag(CapPolicy::Off),
        "both uncapped"
    );
    assert_eq!(tag(CapPolicy::Auto), format!("{v1}/P1W16/Cauto"));
    assert_ne!(tag(CapPolicy::Fixed(1)), tag(CapPolicy::Fixed(2)));
}

/// The base check: the block proof family's base and nothing else — not RPX,
/// not Poseidon1 at another cap, not another hash.
#[test]
fn checked_base_refuses_what_no_block_proves() {
    let family = crate::hash_pin::BLOCK_BASE;
    assert_eq!(family, BaseFormat::P1_WHIR);
    assert!(checked_base(&family).is_ok());
    assert!(
        checked_base(&BaseFormat::RPX).is_err(),
        "RPX left the block path"
    );
    for cap in [
        CapPolicy::Off,
        CapPolicy::Auto,
        CapPolicy::Fixed(1),
        CapPolicy::Fixed(MAX_P1_CAP_HEIGHT as u8),
        CapPolicy::Fixed(MAX_P1_CAP_HEIGHT as u8 + 1),
    ] {
        let base = BaseFormat {
            arity4_cap: cap,
            ..BaseFormat::P1_WHIR
        };
        assert!(checked_base(&base).is_err(), "{cap}");
    }
    for hash in [
        CommitmentHash::Keccak256,
        CommitmentHash::Blake3,
        CommitmentHash::Rpo256,
        CommitmentHash::Rpx256,
        CommitmentHash::Poseidon,
    ] {
        let base = BaseFormat {
            hash,
            ..BaseFormat::P1_WHIR
        };
        assert!(checked_base(&base).is_err(), "{hash:?}");
    }
}

/// ★ A block proves and verifies under the family's base; it verifies under no
/// other (RPX's, or Poseidon1 at another cap), and no other base proves: the
/// format names the base, the family decides it. Groups retire and revive
/// between the phases (drop 3).
#[test]
fn a_block_proves_under_its_family_base_and_no_other_verifies_it() {
    let elf = asm_elf_bytes("all_instructions_64");
    let (rpx, p1) = (
        many_groups(BaseFormat::RPX),
        many_groups(BaseFormat::P1_WHIR),
    );
    let p1_proof = prove(&elf, &p1, 3);
    assert!(p1_proof.proof.columns.len() > 1, "several groups");
    assert!(
        verify(&p1_proof, &elf, &p1),
        "a P1 proof under the P1 format"
    );
    assert!(
        !verify(&p1_proof, &elf, &rpx),
        "a P1 proof under the RPX format"
    );
    assert!(
        prove_block_whir_with(
            &elf,
            &[],
            &ProofOptions::default_test_options(),
            &rpx,
            &options(3),
            &Deviations::default(),
        )
        .is_err(),
        "an RPX base is refused, not proved"
    );
    // The cap is the verifier's constant too.
    let other_cap = many_groups(BaseFormat {
        arity4_cap: CapPolicy::Fixed(1),
        ..BaseFormat::P1_WHIR
    });
    assert!(!verify(&p1_proof, &elf, &other_cap), "C2 read as C1");
}

/// ★ The tag is in the statement: a P1 proof is refused by a verifier that
/// absorbs the untagged `…_V1` tag in its place, and a proof made and verified
/// with that tag in both places is accepted — the tag is the only thing the
/// mutation moves.
#[test]
fn the_p1_statement_tag_is_bound() {
    let elf = asm_elf_bytes("sub");
    let p1 = many_groups(BaseFormat::P1_WHIR);
    let honest = prove(&elf, &p1, 0);
    assert!(verify(&honest, &elf, &p1));
    P1_TAG_OMITTED.with(|omitted| omitted.set(true));
    let refused = !verify(&honest, &elf, &p1);
    let untagged = prove(&elf, &p1, 0);
    let both_untagged = verify(&untagged, &elf, &p1);
    P1_TAG_OMITTED.with(|omitted| omitted.set(false));
    assert!(refused, "a P1 proof verified under the untagged tag");
    assert!(both_untagged, "the mutation is the tag alone");
    assert!(
        !verify(&untagged, &elf, &p1),
        "an untagged proof under the tag"
    );
}

/// ★ The statement the Poseidon1 leaf absorbs call by call
/// (`block_statement_calls`) is `absorb_block`'s own: the calls concatenate to
/// the statement's one run (`block_statement_bytes`, the block id's preimage),
/// and `P1Transcript` fed them lands where `absorb_block` leaves it.
#[test]
fn the_statement_calls_are_absorb_blocks_own() {
    use crypto::fiat_shamir::is_transcript::IsTranscript;
    use crypto::fiat_shamir::p1_transcript::P1Transcript;
    let elf = asm_elf_bytes("sub");
    let p1 = many_groups(BaseFormat::P1_WHIR);
    let proof = prove(&elf, &p1, 0);
    let opts = ProofOptions::default_test_options();
    let statement = proof.statement();
    let config = block_whir::block_frame(statement, &elf, &opts, &p1)
        .expect("the frame")
        .config;
    let tag = block_statement_tag(&p1.zf.base);
    let calls = block_whir::block_statement_calls(statement, &tag, &elf, &config)
        .expect("byte strings only");
    let bytes = block_whir::block_statement_bytes(
        statement,
        &tag,
        &crate::statement::elf_digest(&elf),
        &config,
    );
    assert!(calls.len() > 10, "{} calls", calls.len());
    assert_eq!(calls.concat(), bytes, "the calls are the run, cut");
    let mut direct = P1Transcript::new();
    block_whir::absorb_block(
        &mut direct,
        &tag,
        &elf,
        statement.public_output,
        statement.table_counts,
        statement.num_private_input_pages,
        statement.runtime_page_ranges,
        statement.table_num_vars,
        &config,
        statement.groups,
    );
    let mut replay = P1Transcript::new();
    for call in &calls {
        replay.append_bytes(call);
    }
    assert_eq!(replay.state(), direct.state(), "call for call");
    let mut merged = P1Transcript::new();
    merged.append_bytes(&bytes);
    assert_ne!(
        merged.state(),
        direct.state(),
        "one run is another stream under Poseidon1: the calls are load-bearing"
    );
}

/// The WHIR configuration `base` dispatches to, by its type's name.
fn dispatched(base: BaseFormat) -> Result<&'static str, crate::Error> {
    Ok(crate::with_block_hash!(base, |H| {
        std::any::type_name::<H>()
    }))
}

/// ★ The dispatch itself, whatever the process's knob says: the family's base
/// binds the family's WHIR configuration (`P1Whir`), and every other base —
/// RPX's, keccak's — is refused, not bound to a configuration of its own.
#[test]
fn the_base_alone_binds_the_block_hash() {
    println!(
        "the process's WHIR hash knob: {:?}",
        std::env::var(crate::whir_hash_knob::ENV)
    );
    assert_eq!(
        dispatched(BaseFormat::P1_WHIR).expect("Poseidon1 dispatches"),
        std::any::type_name::<multilinear::whir_hash::P1Whir>()
    );
    assert!(
        dispatched(BaseFormat::RPX).is_err(),
        "an RPX base is refused"
    );
    let keccak = BaseFormat {
        hash: CommitmentHash::Keccak256,
        ..BaseFormat::RPX
    };
    assert!(dispatched(keccak).is_err(), "a keccak base is refused");
}

/// ★ No environment read decides the base: the block's prove, verify and
/// plan sites dispatch through `with_block_hash!` on the format alone, whose
/// body — the base check, then the family's configuration — reads no
/// environment and never consults the WHIR hash knob, and the plan reads no
/// knob either (D-WHIR-P1 S7: `LAMBDA_VM_WHIR_HASH` left the block path).
#[test]
fn no_environment_read_decides_the_base() {
    let block = include_str!("../block_whir.rs");
    assert_eq!(
        block.matches("crate::with_whir_hash!").count(),
        0,
        "every block dispatch is on the format's base"
    );
    assert_eq!(
        block
            .matches("crate::with_block_hash!(format.zf.base")
            .count(),
        3
    );
    let check = &block[block.find("pub fn checked_base").expect("checked_base")..];
    let check = &check[..check.find("\n}\n").expect("its end")];
    let tag = &block[block.find("pub fn block_statement_tag").expect("the tag")..];
    let tag = &tag[..tag.find("\n}\n").expect("its end")];
    let knob = include_str!("../whir_hash_knob.rs");
    let arms = &knob[knob
        .find("macro_rules! with_block_hash")
        .expect("the macro")..];
    let arms = &arms[..arms.find("\n}\n").expect("its end")];
    assert!(
        arms.contains("checked_base(&$base)?")
            && arms.contains("type $h = $crate::hash_pin::BlockWhirHash;"),
        "the macro checks the base and binds the family's configuration"
    );
    for (name, source) in [
        ("checked_base", check),
        ("the tag", tag),
        ("the macro's arms", arms),
    ] {
        for read in [
            "std::env",
            "env::var",
            "selected()",
            "whir_hash_knob",
            "with_whir_hash",
        ] {
            assert!(!source.contains(read), "{name} names {read}");
        }
    }
    // The plan reads other knobs (`LFM_WHIR_SHARE_INVERSE`), never the hash's.
    let plan = include_str!("../lfm/whir_block.rs");
    for read in ["whir_hash_knob", "with_whir_hash", "LAMBDA_VM_WHIR_HASH"] {
        assert!(!plan.contains(read), "the block plan names {read}");
    }
}

// ==================== the block path's source scan ====================

/// What a scan of the block path reads: production code as tokens, with every
/// `#[cfg(test)]` item and statement left out, comments dropped and the
/// contents of string and character literals blanked. A name in a comment, a
/// doc or a string is no site; a name in code is one, however it is spelled
/// across whitespace (REV-P3B14-A A3: a legacy prove left in code beside the
/// family's name in a comment passed the text scan this replaces).
mod scan {
    /// `src` with comments dropped and string / char literal contents blanked
    /// (a literal becomes `""` or `' '`; a lifetime stays). Newlines kept.
    fn masked(src: &str) -> String {
        let c: Vec<char> = src.chars().collect();
        let mut out = String::with_capacity(src.len());
        let mut i = 0;
        let ident = |ch: char| ch.is_ascii_alphanumeric() || ch == '_';
        while i < c.len() {
            let next = c.get(i + 1).copied();
            if c[i] == '/' && next == Some('/') {
                while i < c.len() && c[i] != '\n' {
                    i += 1;
                }
            } else if c[i] == '/' && next == Some('*') {
                let mut depth = 0usize;
                while i < c.len() {
                    if c[i] == '/' && c.get(i + 1) == Some(&'*') {
                        depth += 1;
                        i += 2;
                    } else if c[i] == '*' && c.get(i + 1) == Some(&'/') {
                        depth -= 1;
                        i += 2;
                        if depth == 0 {
                            break;
                        }
                    } else {
                        if c[i] == '\n' {
                            out.push('\n');
                        }
                        i += 1;
                    }
                }
            } else if c[i] == 'r'
                && (i == 0 || !ident(c[i - 1]))
                && matches!(next, Some('"') | Some('#'))
                && {
                    let mut j = i + 1;
                    while c.get(j) == Some(&'#') {
                        j += 1;
                    }
                    c.get(j) == Some(&'"')
                }
            {
                let mut j = i + 1;
                let mut hashes = 0;
                while c[j] == '#' {
                    hashes += 1;
                    j += 1;
                }
                j += 1;
                while j < c.len()
                    && !(c[j] == '"' && (0..hashes).all(|k| c.get(j + 1 + k) == Some(&'#')))
                {
                    if c[j] == '\n' {
                        out.push('\n');
                    }
                    j += 1;
                }
                out.push_str("\"\"");
                i = j + 1 + hashes;
            } else if c[i] == '"' {
                let mut j = i + 1;
                while j < c.len() && c[j] != '"' {
                    if c[j] == '\\' {
                        j += 1;
                    }
                    if c.get(j) == Some(&'\n') {
                        out.push('\n');
                    }
                    j += 1;
                }
                out.push_str("\"\"");
                i = j + 1;
            } else if c[i] == '\'' && next == Some('\\') {
                let mut j = i + 2;
                while j < c.len() && c[j] != '\'' {
                    j += 1;
                }
                out.push_str("' '");
                i = j + 1;
            } else if c[i] == '\'' && c.get(i + 2) == Some(&'\'') {
                out.push_str("' '");
                i += 3;
            } else {
                out.push(c[i]);
                i += 1;
            }
        }
        out
    }

    /// `src` tokenised: identifiers and numbers as words, `::` as one token,
    /// every other non-space character a token of its own.
    pub(super) fn tokens(src: &str) -> Vec<String> {
        let c: Vec<char> = src.chars().collect();
        let mut out = Vec::new();
        let mut i = 0;
        while i < c.len() {
            if c[i].is_whitespace() {
                i += 1;
            } else if c[i].is_ascii_alphanumeric() || c[i] == '_' {
                let start = i;
                while i < c.len() && (c[i].is_ascii_alphanumeric() || c[i] == '_') {
                    i += 1;
                }
                out.push(c[start..i].iter().collect());
            } else if c[i] == ':' && c.get(i + 1) == Some(&':') {
                out.push("::".to_string());
                i += 2;
            } else {
                out.push(c[i].to_string());
                i += 1;
            }
        }
        out
    }

    /// The production tokens of a source: its masked tokens with every
    /// `#[cfg(test)]`-gated item or statement removed — from the attribute to
    /// the end of what it gates. An ITEM (`fn`, `struct`, `impl`, `mod`,
    /// `const`, `use`, … after any visibility) ends at the first `;` at its own
    /// depth or at the close of the block it opens; a comma never ends one, so a
    /// generic `Result<A, B>` in its signature is no end. Anything else (a
    /// field, a variant, a match arm, a statement) ends at the first `;` or `,`
    /// at its own depth or at the close of a block it opens — a comma inside a
    /// generic type ends it early, which scans more code, never less. Either
    /// way the close of the enclosing block ends it. No `<` is read as a
    /// bracket: in an arm or a field it may be a comparison or a shift.
    pub(super) fn production(src: &str) -> Vec<String> {
        let t = tokens(&masked(src));
        let gate: Vec<String> = tokens("#[cfg(test)]");
        let mut out = Vec::with_capacity(t.len());
        let mut i = 0;
        while i < t.len() {
            if t[i..].starts_with(&gate) {
                i += gate.len();
                // Further attributes on the same item.
                while t.get(i).map(String::as_str) == Some("#") {
                    let mut depth = 0i64;
                    i += 1;
                    while i < t.len() {
                        match t[i].as_str() {
                            "[" => depth += 1,
                            "]" => {
                                depth -= 1;
                                if depth == 0 {
                                    i += 1;
                                    break;
                                }
                            }
                            _ => {}
                        }
                        i += 1;
                    }
                }
                // Past any visibility (`pub`, `pub(crate)`, …): an item keyword?
                let mut k = i;
                if t.get(k).map(String::as_str) == Some("pub") {
                    k += 1;
                    if t.get(k).map(String::as_str) == Some("(") {
                        while k < t.len() && t[k] != ")" {
                            k += 1;
                        }
                        k += 1;
                    }
                }
                let item = matches!(
                    t.get(k).map(String::as_str),
                    Some(
                        "fn" | "struct"
                            | "enum"
                            | "union"
                            | "impl"
                            | "trait"
                            | "mod"
                            | "type"
                            | "const"
                            | "static"
                            | "use"
                            | "extern"
                            | "macro_rules"
                            | "unsafe"
                            | "async"
                    )
                );
                let mut depth = 0i64;
                while i < t.len() {
                    match t[i].as_str() {
                        "{" | "(" | "[" => depth += 1,
                        "}" | ")" | "]" => {
                            if depth == 0 {
                                // The enclosing block closes: the gated item ended.
                                break;
                            }
                            depth -= 1;
                            if depth == 0 && t[i] == "}" {
                                i += 1;
                                if matches!(t.get(i).map(String::as_str), Some(";") | Some(",")) {
                                    i += 1;
                                }
                                break;
                            }
                        }
                        ";" if depth == 0 => {
                            i += 1;
                            break;
                        }
                        "," if depth == 0 && !item => {
                            i += 1;
                            break;
                        }
                        _ => {}
                    }
                    i += 1;
                }
                continue;
            }
            out.push(t[i].clone());
            i += 1;
        }
        out
    }

    /// Occurrences of `pattern`'s tokens as a contiguous run in `code`.
    pub(super) fn count(code: &[String], pattern: &str) -> usize {
        let p = tokens(pattern);
        code.windows(p.len()).filter(|w| *w == p.as_slice()).count()
    }

    /// The legacy pin's names and entry points, as code: none may appear in a
    /// production line of the block path. Whole identifiers (so
    /// `harvest_child_under` is not `harvest_child`), and the legacy family's
    /// `LfmFilled::prove` / `prove` method as a call (`.prove(`, `::prove(`).
    pub(super) const LEGACY: &[&str] = &[
        "LegacyFamily",
        "BlockStarkHash",
        "BlockTranscript",
        "block_transcript",
        "BlockProver",
        "BlockVerifier",
        "BLOCK_HASHER",
        "BLOCK_COMMITMENT_HASH",
        "WrapHash::production",
        "verify_against_artifacts",
        "verify_against_chunked",
        "harvest_child",
        "harvest_child_verified",
        "host_table_forked",
        "lfm_prove",
        "lfm_prove_with_hasher",
        "prove_traces_with_hasher",
        "build_artifacts_counted",
        "build_artifacts_with_hasher",
        "registry::build_artifacts",
        "from_artifacts(",
        "TableVerifyShape::derive(",
        ".prove(",
        "::prove(",
    ];

    /// What the block path takes from its family alone, as code: a binary
    /// tree shape (the block's shapes are `TreeGeometry::of::<Fam::Stark>()`),
    /// and the Poseidon1 family or its configuration module by name (the
    /// block names `BlockFamily`, so a flip of that alias moves every site).
    /// None may appear in a production line of the block path.
    pub(super) const FAMILY_ONLY: &[&str] = &["TreeGeometry::BINARY", "P1Family", "p1_commit"];

    /// One file of the scan: its name, its production entry points (each must
    /// survive the `#[cfg(test)]` skipper, so a misplaced gate cannot hide the
    /// code after it), and the family-instantiated sites it must hold, with
    /// their exact counts.
    pub(super) struct File {
        pub name: &'static str,
        pub source: String,
        pub anchors: &'static [&'static str],
        pub sites: &'static [(&'static str, usize)],
        /// Names this file must not use beyond [`LEGACY`].
        pub forbidden: &'static [&'static str],
    }

    /// The scan: `Err` names the first violation.
    pub(super) fn check(files: &[File]) -> Result<(), String> {
        for f in files {
            let code = production(&f.source);
            if code.len() < 1000 {
                return Err(format!("{}: {} production tokens read", f.name, code.len()));
            }
            for a in f.anchors {
                if count(&code, a) == 0 {
                    return Err(format!("{}: the scan does not reach `{a}`", f.name));
                }
            }
            for name in LEGACY.iter().chain(FAMILY_ONLY).chain(f.forbidden) {
                let n = count(&code, name);
                if n > 0 {
                    return Err(format!(
                        "{} names `{name}` in production code ({n}×)",
                        f.name
                    ));
                }
            }
            for (site, want) in f.sites {
                let n = count(&code, site);
                if n != *want {
                    return Err(format!(
                        "{}: `{site}` {n}× in production code, want {want}",
                        f.name
                    ));
                }
            }
        }
        Ok(())
    }

    /// The block path's production files, as built.
    pub(super) fn block_path() -> Vec<File> {
        vec![
            File {
                name: "block_whir.rs",
                source: include_str!("../block_whir.rs").to_string(),
                anchors: &[
                    "pub fn checked_base(",
                    "pub fn block_statement_tag(",
                    "pub fn prove_block_whir(",
                    "pub fn verify_block_whir(",
                    "fn verify_block_whir_with(",
                ],
                sites: &[],
                forbidden: &[],
            },
            File {
                name: "lfm/whir_block.rs",
                source: include_str!("../lfm/whir_block.rs").to_string(),
                anchors: &[
                    "fn derive_capped(",
                    "pub fn node_program(",
                    "pub fn artifacts_of(",
                    "fn derived_of(",
                    "pub fn verify_block_tree(",
                    "fn verify_block_tree_with(",
                ],
                forbidden: &[],
                sites: &[
                    ("build_artifacts_counted_under::<BlockStark>(", 1),
                    ("from_artifacts_under::<crate::hash_pin::BlockFamily>(", 1),
                    (
                        "verify_against_artifacts_under::<crate::hash_pin::BlockFamily>(",
                        1,
                    ),
                    ("node_program_under::<crate::hash_pin::BlockFamily>(", 1),
                ],
            },
            File {
                name: "lfm/whir_block_tree.rs",
                source: include_str!("../lfm/whir_block_tree.rs").to_string(),
                anchors: &[
                    "fn build_nodes(",
                    "fn prove_tree_pipelined(",
                    "pub fn prove_whir_block_tree(",
                    "pub fn verify_whir_block_tree_proof(",
                ],
                forbidden: &[],
                sites: &[
                    ("prove_under::<BlockFamily>(", 2),
                    ("harvest_child_under::<BlockFamily>(", 2),
                ],
            },
            File {
                name: "lfm/block_node.rs",
                source: include_str!("../lfm/block_node.rs").to_string(),
                anchors: &["pub fn emit_block_node(", "fn emit_node("],
                sites: &[],
                forbidden: &[],
            },
            File {
                name: "lfm/harvest.rs",
                source: include_str!("../lfm/harvest.rs").to_string(),
                anchors: &[
                    "fn build_table_legs_with(",
                    "fn harvest_child_under<",
                    "fn try_child_arena_words(",
                ],
                sites: &[],
                // The child's arenas are laid at its family's digest width,
                // never the legacy pin's.
                forbidden: &["words_per_root", "commitments_to_arena", "commitment_words"],
            },
        ]
    }
}

/// ★ The block path names one proof family (D-P3B §2.1, review item 1): no
/// production line of the plan, the tree driver, the node or the block's
/// dispatch names the legacy family, the legacy pin's configuration,
/// transcript, prover, verifier or socket, or a legacy-family entry point
/// (`LegacyFamily`'s instances: `lfm_prove`, `LfmFilled::prove`,
/// `verify_against_artifacts`, `harvest_child`, …); every LFM site of the W3
/// tree is instantiated at `BlockFamily`, at its exact count. Nor does one
/// name a binary tree shape, the Poseidon1 family or its configuration module
/// (`scan::FAMILY_ONLY`): the block takes its shapes and its hash from
/// `BlockFamily` alone. Read as code tokens ([`scan`]), so a comment or a
/// string neither hides a site nor fakes one.
#[test]
fn no_block_path_site_names_the_legacy_pin() {
    scan::check(&scan::block_path()).unwrap_or_else(|e| panic!("{e}"));
}

/// ★ The scan refuses the regressions it exists for, planted into the real
/// sources (REV-P3B14-A A3): the leaf prove put back to dcc4703e9's legacy
/// `filled.prove(artifacts, …)` with the family's name left in a comment; a
/// legacy name in a statement split over lines; a `#[cfg(test)]` that would
/// swallow a production function; a binary tree shape at the harvest's
/// family site, the Poseidon1 family named in the plan in place of
/// `BlockFamily`, and its configuration module named there; and the scan
/// reads past comments and strings without false positives.
#[test]
fn the_block_path_scan_refuses_planted_regressions() {
    let planted = |file: &str, from: &str, to: &str| -> Result<(), String> {
        let mut files = scan::block_path();
        let f = files
            .iter_mut()
            .find(|f| f.name == file)
            .expect("a scanned file");
        assert_eq!(f.source.matches(from).count(), 1, "{file}: `{from}` once");
        f.source = f.source.replacen(from, to, 1);
        scan::check(&files)
    };
    assert_eq!(scan::check(&scan::block_path()), Ok(()));
    // dcc4703e9's leaf prove, the family's call left in a comment.
    let leaf = "filled.prove_under::<BlockFamily>(artifacts, &wrap, decide_lfm_residency());";
    let legacy_leaf = "filled.prove(artifacts, &wrap, decide_lfm_residency()); \
                       // filled.prove_under::<BlockFamily>(artifacts, &wrap, decide_lfm_residency());";
    let e =
        planted("lfm/whir_block_tree.rs", leaf, legacy_leaf).expect_err("the legacy leaf prove");
    assert!(e.contains("`.prove(`"), "{e}");
    // A legacy entry point spelled across lines and spaces.
    let e = planted(
        "lfm/whir_block.rs",
        "pub fn artifacts_of(",
        "fn planted() { super :: proof ::\n    verify_against_artifacts (); }\npub fn artifacts_of(",
    )
    .expect_err("a split legacy call");
    assert!(e.contains("`verify_against_artifacts`"), "{e}");
    // A test gate that would hide the verifier's entry point.
    let e = planted(
        "lfm/whir_block_tree.rs",
        "pub fn verify_whir_block_tree_proof(",
        "#[cfg(test)]\npub fn verify_whir_block_tree_proof(",
    )
    .expect_err("a gated entry point");
    assert!(e.contains("does not reach"), "{e}");
    // A binary shape where the harvest takes its family's.
    let e = planted(
        "lfm/harvest.rs",
        "stark::config::TreeGeometry::of::<Fam::Stark>(),",
        "stark::config::TreeGeometry::BINARY,",
    )
    .expect_err("a binary shape on the block path");
    assert!(e.contains("`TreeGeometry::BINARY`"), "{e}");
    // The Poseidon1 family by name where the plan names the block's.
    let e = planted(
        "lfm/whir_block.rs",
        "from_artifacts_under::<crate::hash_pin::BlockFamily>(",
        "from_artifacts_under::<crate::hash_pin::P1Family>(",
    )
    .expect_err("the Poseidon1 family by name");
    assert!(e.contains("`P1Family`"), "{e}");
    // Its configuration module named in the plan.
    let e = planted(
        "lfm/whir_block.rs",
        "pub fn artifacts_of(",
        "fn planted() -> u8 { crate::lfm::p1_commit::P1_STARK_CAP }\npub fn artifacts_of(",
    )
    .expect_err("the Poseidon1 configuration module by name");
    assert!(e.contains("`p1_commit`"), "{e}");
    // Comments and strings are no sites.
    assert_eq!(
        planted(
            "lfm/block_node.rs",
            "pub fn emit_block_node(",
            "// LegacyFamily, harvest_child(, .prove(\nconst _PLANTED: &str = \"lfm_prove( BLOCK_HASHER\";\n\
             /* verify_against_artifacts( */\npub fn emit_block_node(",
        ),
        Ok(())
    );
    // A test-gated statement inside a production function stays out.
    assert_eq!(
        scan::count(
            &scan::production("fn f() {\n    #[cfg(test)]\n    lfm_prove(a, b);\n    g();\n}\n"),
            "lfm_prove"
        ),
        0
    );
    assert_eq!(
        scan::count(
            &scan::production("fn f() {\n    lfm_prove(a, b);\n}\n"),
            "lfm_prove"
        ),
        1
    );
    // A gated function whose signature holds a generic comma is left out
    // whole, its body included.
    assert_eq!(
        scan::count(
            &scan::production(
                "#[cfg(test)]\nfn t(a: u8) -> Result<u8, String> {\n    lfm_prove(a);\n}\nfn g() {}\n"
            ),
            "lfm_prove"
        ),
        0
    );
    // A gated struct-literal field or match arm holding a shift or a
    // comparison ends at its comma: the fields and arms after it are scanned.
    assert_eq!(
        scan::count(
            &scan::production(
                "fn f() -> S {\n    S {\n        #[cfg(test)]\n        a: 1 << 2,\n        b: lfm_prove,\n    }\n}\n"
            ),
            "lfm_prove"
        ),
        1
    );
    assert_eq!(
        scan::count(
            &scan::production(
                "fn f(k: K) -> bool {\n    match k {\n        #[cfg(test)]\n        K::A => x < y,\n        K::B => lfm_prove(),\n    }\n}\n"
            ),
            "lfm_prove"
        ),
        1
    );
    // A gated item whose signature carries a where-clause comma is left out
    // whole, to its block's close.
    assert_eq!(
        scan::count(
            &scan::production(
                "#[cfg(test)]\npub(crate) fn t<P>(p: P) -> u8\nwhere\n    P: A,\n    u8: B<P>,\n{\n    lfm_prove(p)\n}\nfn g() { lfm_prove(1) }\n"
            ),
            "lfm_prove"
        ),
        1
    );
    // A gated field of a generic type ends at its comma.
    assert_eq!(
        scan::count(
            &scan::production(
                "struct S {\n    #[cfg(test)]\n    a: Result<u8, String>,\n    b: lfm_prove,\n}\n"
            ),
            "lfm_prove"
        ),
        1
    );
}
