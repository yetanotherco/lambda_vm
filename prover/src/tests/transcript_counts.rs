/// The Fiat-Shamir transcript's state machine, counted rather than run.
///
/// Mirrors `DefaultTranscript` exactly: an absorb is one `Update::update` and
/// invalidates the duplex output buffer; a squeeze is one `finalize_reset`
/// PLUS the chain-advance `update` of the 32 reversed bytes, which is itself a
/// counted absorb; `state()` is a finalize on a clone that neither resets nor
/// re-absorbs. Four candidates come out of one squeeze, an extension element
/// takes three, a `sample_u64` takes one.
#[derive(Default, Debug, Clone, Copy, PartialEq, Eq)]
pub struct TranscriptCounts {
    pub absorb_calls: u64,
    pub absorb_bytes: u64,
    pub squeezes: u64,
    pub state_finalizes: u64,
}

impl TranscriptCounts {
    /// Every finalize the transcript performs: one per squeeze, one per state.
    pub fn finalizes(&self) -> u64 {
        self.squeezes + self.state_finalizes
    }

    /// `transcript_absorbs` as A2a counts it: the APPEND calls only.
    ///
    /// `absorb_calls` here includes the chaining re-absorb a squeeze performs,
    /// which A2a attributes to squeezing rather than to an absorb anyone asked
    /// for. Subtracting it is A2a's own non-vacuity identity read backwards:
    /// `absorb_calls == transcript_absorbs + transcript_squeezes`.
    pub fn transcript_absorbs(&self) -> u64 {
        self.absorb_calls - self.squeezes
    }

    /// `transcript_squeezes` as A2a counts it: one per `sample()`.
    pub fn transcript_squeezes(&self) -> u64 {
        self.squeezes
    }
}

struct Sim {
    out_pos: usize,
    c: TranscriptCounts,
}

const SQUEEZE_LEN: usize = 32;

impl Sim {
    /// `DefaultTranscript::new(&[])` absorbs the empty seed, which is still one
    /// `update` call of zero bytes.
    fn new() -> Self {
        let mut s = Self {
            out_pos: SQUEEZE_LEN,
            c: TranscriptCounts::default(),
        };
        s.absorb(0);
        s
    }
    fn absorb(&mut self, n: u64) {
        self.out_pos = SQUEEZE_LEN;
        self.c.absorb_calls += 1;
        self.c.absorb_bytes += n;
    }
    fn squeeze(&mut self) {
        self.c.squeezes += 1;
        // `sample()` re-absorbs its own reversed output to advance the chain.
        self.c.absorb_calls += 1;
        self.c.absorb_bytes += 32;
        self.out_pos = 0;
    }
    fn next_u64(&mut self) {
        if self.out_pos + 8 > SQUEEZE_LEN {
            self.squeeze();
        }
        self.out_pos += 8;
    }
    /// A cubic-extension element: three base coordinates, each one candidate.
    /// Exact unless a candidate lands at or above the modulus, which is one
    /// extra draw with probability about 2^-32 per coordinate.
    fn sample_ext(&mut self) {
        for _ in 0..3 {
            self.next_u64();
        }
    }
    fn sample_u64(&mut self) {
        self.next_u64();
    }
    fn state(&mut self) {
        self.c.state_finalizes += 1;
    }
}

/// One table's transcript shape, read off the AIR and its layout.
#[derive(Clone, Copy, Debug)]
pub struct TableTranscriptShape {
    /// Height in variables.
    pub n: usize,
    /// GKR input-layer variables, `n + ceil_log2(interactions)`.
    pub m: usize,
    /// The batched sumcheck's degree, `max(shape.degree() + 1, 2)`.
    pub degree: usize,
    /// Committed factors, whose values `claim_reduce` absorbs.
    pub factors: usize,
    /// Main columns, whose values `claim_reduce` absorbs at the end.
    pub columns: usize,
}

/// One commitment group: the stack it landed on and how many columns it holds.
#[derive(Clone, Copy, Debug)]
pub struct GroupTranscriptShape {
    pub n_stack: usize,
    pub num_polys: usize,
    pub columns: usize,
}

fn drive_table(s: &mut Sim, t: &TableTranscriptShape) {
    // multilinear_table::verify
    s.absorb(24);
    s.absorb(24); // bus_output.0, .1
    // gkr::verify: m layers, layer i runs i degree-3 sumcheck rounds
    for i in 0..t.m {
        s.sample_ext(); // lambda
        for _ in 0..i {
            for _ in 0..3 {
                s.absorb(24);
            }
            s.sample_ext();
        }
        for _ in 0..4 {
            s.absorb(24); // p_lo, p_hi, q_lo, q_hi
        }
        s.sample_ext(); // c
    }
    for _ in 0..t.n {
        s.sample_ext(); // r
    }
    // constraint_argument::verify_core -> batch::verify
    for _ in 0..3 {
        s.absorb(24); // the three rule claims
    }
    s.sample_ext(); // the batching challenge
    for _ in 0..t.n {
        for _ in 0..t.degree {
            s.absorb(24);
        }
        s.sample_ext();
    }
    // claim_reduce::verify
    for _ in 0..t.factors {
        s.absorb(24);
    }
    s.sample_ext();
    for _ in 0..t.n {
        for _ in 0..2 {
            s.absorb(24);
        }
        s.sample_ext();
    }
    for _ in 0..t.columns {
        s.absorb(24);
    }
}

/// Today's uniform fold schedule, spelled independently of `ChainConfig`: `k`
/// per round, the remainder last. The LEGACY format's schedule; the closed
/// form below drives the config's own schedule, and
/// `the_uniform_schedule_is_the_legacy_configs` ties the two at the legacy
/// format.
fn schedule(num_vars: usize, k: usize) -> Vec<usize> {
    let mut out = Vec::new();
    let mut left = num_vars;
    while left > 0 {
        let take = k.min(left);
        out.push(take);
        left -= take;
    }
    out
}

/// One chain's transcript over the fold schedule `sch` — the config's own
/// (`ChainConfig::schedule`), so a non-uniform first fold (`whir_folds=first6`,
/// the production default) is priced as it is proved — and its grinds, of
/// which only those with bits touch the transcript (P2's query-only grind
/// spends one a round).
fn drive_chain(s: &mut Sim, sch: &[usize], queries: usize, grind: GrindBits) {
    // `check_grind`: at zero bits it returns before its `state()` and before
    // absorbing the nonce.
    let spend = |s: &mut Sim, bits: u8| {
        if bits > 0 {
            s.state();
            s.absorb(8);
        }
    };
    let rounds = sch.len();
    for (r, &kr) in sch.iter().enumerate() {
        spend(s, grind.folding);
        for _ in 0..kr {
            s.absorb(24);
            s.absorb(24); // degree-2 round polynomial
            s.sample_ext();
        }
        if r + 1 < rounds {
            s.absorb(32); // next_root
            s.sample_ext(); // z0
            s.absorb(24); // y0
            spend(s, grind.ood);
            s.sample_ext(); // gamma
            spend(s, grind.query);
        } else {
            s.absorb(24); // final_value
            spend(s, grind.query);
        }
        for _ in 0..queries {
            s.sample_u64();
        }
    }
}

/// One whole proof's transcript: the statement, the roots, every table, every
/// commitment group's opening.
#[allow(clippy::too_many_arguments)]
pub fn transcript_counts(
    statement_absorbs: &[u64],
    tables: &[TableTranscriptShape],
    groups: &[GroupTranscriptShape],
    chain: &multilinear::whir_chain::ChainConfig,
    queries: usize,
    owed_probe: bool,
) -> TranscriptCounts {
    let mut s = Sim::new();
    for &n in statement_absorbs {
        s.absorb(n);
    }
    let roots: usize = groups.iter().map(|g| g.num_polys).sum();
    // `owed` runs the COMMIT bus's counterparty on a CLONE of the transcript:
    // every root again, then z and alpha. The clone is thrown away, so it moves
    // no challenge — but it is real hashing and a counter sees it. Only the
    // epoch path does this; the cross-epoch proof has no commit bus to owe.
    if owed_probe {
        let mut probe = Sim {
            out_pos: SQUEEZE_LEN,
            c: TranscriptCounts::default(),
        };
        for _ in 0..roots {
            probe.absorb(32);
        }
        probe.sample_ext();
        probe.sample_ext();
        s.c.absorb_calls += probe.c.absorb_calls;
        s.c.absorb_bytes += probe.c.absorb_bytes;
        s.c.squeezes += probe.c.squeezes;
    }
    // multi_verify: every root, then z, alpha, beta.
    for _ in 0..roots {
        s.absorb(32);
    }
    for _ in 0..3 {
        s.sample_ext();
    }
    for t in tables {
        drive_table(&mut s, t);
    }
    for g in groups {
        for _ in 0..g.columns {
            s.absorb(24);
        }
        s.sample_ext(); // the batching challenge
        for _ in 0..g.num_polys {
            drive_chain(&mut s, &chain.schedule(g.n_stack), queries, chain.grind);
        }
    }
    s.c
}

// =========================================================================
// Driving it off the real AIRs
// =========================================================================

use multilinear::whir_chain::GrindBits;

/// A grind with bits reads the transcript's state once (`check_grind`); one of
/// zero bits reads nothing. So a production chain reads it `R` times under the
/// default (`whir_grind=query`: the query grind alone) and `3R − 1` times under
/// the opt-out (`whir_grind=all`).
#[test]
fn a_chain_reads_the_state_once_a_spent_grind() {
    let sch = [6, 4, 4, 4, 4, 3];
    let states = |grind| {
        let mut s = Sim::new();
        drive_chain(&mut s, &sch, 112, grind);
        s.c.state_finalizes
    };
    assert_eq!(states(GrindBits::query_only(20)), 6);
    assert_eq!(states(GrindBits::uniform(20)), 17);
    assert_eq!(states(GrindBits::default()), 0);
    for (whir_grind, want) in [
        (crate::zf_format::WhirGrind::Query, 6),
        (crate::zf_format::WhirGrind::All, 17),
    ] {
        let config = crate::multilinear_prove::chain_config_under(
            &crate::zf_format::ZfFormat {
                whir_grind,
                ..crate::zf_format::ZfFormat::DEFAULT
            },
            &[(1, 25)],
        );
        assert_eq!(config.schedule(25), sch);
        assert_eq!(states(config.grind), want, "{whir_grind}");
    }
}

/// The closed form drives `ChainConfig::schedule`; at the LEGACY format that
/// is exactly the uniform schedule spelled independently above, at every
/// height a chain reaches. Under the production default (first6) the two
/// differ, which is why the form no longer takes a fold width.
#[test]
fn the_uniform_schedule_is_the_legacy_configs() {
    let legacy = crate::multilinear_prove::chain_config_under(
        &crate::zf_format::ZfFormat::LEGACY,
        &[(1, 25)],
    );
    for n in 1..=32 {
        assert_eq!(
            schedule(n, legacy.log_folding),
            legacy.schedule(n),
            "n = {n}"
        );
    }
    let production = crate::multilinear_prove::chain_config_under(
        &crate::zf_format::ZfFormat::DEFAULT,
        &[(1, 25)],
    );
    assert_ne!(
        schedule(25, production.log_folding),
        production.schedule(25),
        "first6 is not the uniform walk"
    );
}
