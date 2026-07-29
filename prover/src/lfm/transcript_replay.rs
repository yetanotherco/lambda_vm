//! `TranscriptReplay` — the production `DefaultTranscript` replayed inside the
//! machine.
//!
//! This is an eDSL library, not a chip: it is ordinary Rust that tracks the
//! transcript's state AT EMIT TIME and emits the instructions that reproduce
//! the transcript's VALUES at run time. The split matters. Which squeeze a
//! challenge comes from, where a refill lands, which absorb invalidates the
//! output buffer — all of that is decided by the emitter and baked into the
//! program's shape. Only the field arithmetic and the keccak rows are machine
//! work.
//!
//! The mirror it must match is `crypto::fiat_shamir::default_transcript`
//! (post-#841): a keccak sponge with a Plonky3-style duplex output buffer.
//! [`super::keccak_host::TranscriptModel`] is the host model of the same state
//! machine and is checked against the real thing in `machine_tests`; this type
//! tracks the identical `segment` / `out_pos` pair.
//!
//! ## What makes the replay cheap
//!
//! Two identities do all the work:
//!
//! 1. **The reversal cancels.** `sample()` returns the digest byte-REVERSED, and
//!    candidates are read big-endian out of those reversed bytes. The two
//!    reversals cancel exactly: candidate `i` is the PLAIN digest's `u64` lane
//!    `3 − i` (see [`super::keccak_host::candidate_from_state`]). So sampling
//!    never reverses anything — it reads `u32` halves straight off the keccak
//!    state words, two `Unpack`s per squeeze. The reversed digest is emitted
//!    only for the RE-ABSORB (and for a raw [`TranscriptReplay::sample`], whose
//!    return value *is* those bytes).
//!
//! 2. **Canonicity is one instruction.** `p = (2^32 − 1)·2^32 + 1`, so a
//!    candidate `hi·2^32 + lo` is out of range exactly when
//!    `hi = 2^32 − 1 ∧ lo ≠ 0` — and `div` is constrained as `OUT·B = A`, which
//!    is provable with `B = 0` only when `A = 0`. See [`assert_canonical`].
//!
//! A third property is about the emitter rather than the machine: the segment is
//! packed into `u32` halves per SEGMENT, never per append, which is what makes
//! constants of arbitrary length safe anywhere in the stream. The argument is in
//! [`TranscriptReplay::append_const_bytes`] and should be read before touching
//! the append path.
//!
//! ## The zero-rejection restriction
//!
//! A straight-line program has one shape. The production sampler rejects
//! out-of-range candidates and draws again, so the number of candidates a draw
//! consumes — and therefore every later draw's buffer position — is
//! DATA-DEPENDENT. A machine with no branches cannot follow that. The emitted
//! program therefore encodes the no-rejection schedule and is unprovable for the
//! (vanishingly rare) transcript that rejects. See `SOUNDNESS.md` §6.3 for the
//! completeness bound, for why this costs COMPLETENESS only — a rejecting
//! transcript yields no proof, never a wrong one — and for why supporting one
//! rejection is NOT an emitter parameter but a change to the production
//! sampler.

use crate::tables::types::FE;

use super::builder::{Bit, Cell, Ext, Felt, LfmBuilder};
use super::edsl;
use super::keccak_host::{BYTES_PER_HALF, SQUEEZE_LEN};
use super::layout::keccak::DIGEST_WORDS;

/// `u32` halves in one 32-byte squeeze.
const SQUEEZE_HALVES: usize = SQUEEZE_LEN / BYTES_PER_HALF;

/// Bytes in one 64-bit candidate.
const CANDIDATE_BYTES: usize = 8;

/// Candidates one squeeze yields.
const CANDIDATES_PER_SQUEEZE: usize = SQUEEZE_LEN / CANDIDATE_BYTES;

/// `2^32 − 1` — the only `hi` half that can put a candidate at or above `p`.
const HI_MAX: u64 = 0xFFFF_FFFF;

/// A piece of the pending segment, held UNPACKED until the squeeze.
///
/// Packing is per SEGMENT, never per append — see
/// [`TranscriptReplay::append_const_bytes`] for why that distinction is the
/// whole design.
enum SegPiece {
    /// Compile-time bytes. Consecutive runs of these are concatenated before
    /// being chunked into halves, so a constant of any length may sit anywhere.
    Const(Vec<u8>),
    /// Machine-computed `u32` halves, four bytes each little-endian. Opaque
    /// felts, so they must land on a 4-byte boundary of the segment.
    Halves(Vec<Felt>),
    /// A machine half carrying only its low `n` bytes (`n` in `1..4`) — the
    /// trailing piece of a byte string whose length is not a multiple of four.
    /// The packer masks it and pins the unused high bytes to zero.
    Partial(Felt, usize),
}

/// A 64-bit candidate as the two `u32` halves the machine actually holds:
/// `value = hi·2^32 + lo`.
#[derive(Debug, Clone, Copy)]
pub struct Candidate {
    pub lo: Felt,
    pub hi: Felt,
}

/// The squeeze currently backing the output buffer.
///
/// Held as the PLAIN digest's two words with their lane unpacks memoized:
/// candidates 0 and 1 live in word 1 and candidates 2 and 3 in word 0, so a
/// draw that consumes one or two candidates emits a single `Unpack`.
struct SqueezeBuf {
    words: [Cell; DIGEST_WORDS],
    lanes: [Option<[Felt; 4]>; DIGEST_WORDS],
}

/// Emit-time replay of `DefaultTranscript`.
pub struct TranscriptReplay {
    /// The pending segment — the hasher's unfinalized input — as unpacked
    /// pieces. Packed into halves at squeeze time, not at append time.
    segment: Vec<SegPiece>,
    /// The segment's length in BYTES: what drives keccak's length-dependent
    /// padding, and what decides where every half boundary falls.
    segment_len: usize,
    buf: Option<SqueezeBuf>,
    /// Bytes already handed out of the buffer; `SQUEEZE_LEN` means "empty, the
    /// next candidate forces a squeeze".
    out_pos: usize,
}

impl TranscriptReplay {
    /// `DefaultTranscript::new(seed)` — an empty sponge with `seed` absorbed.
    ///
    /// The seed is a program constant, interned by the builder. Anything the
    /// machine COMPUTES is absorbed with [`TranscriptReplay::append_halves`].
    pub fn new(seed: &[u8]) -> Self {
        let mut t = Self {
            segment: Vec::new(),
            segment_len: 0,
            buf: None,
            out_pos: SQUEEZE_LEN,
        };
        t.append_const_bytes(seed);
        t
    }

    /// Absorb machine-computed data: `4 · halves.len()` bytes, four per half,
    /// little-endian.
    ///
    /// Whole halves only, and there is no length parameter. A half is four
    /// consecutive bytes of the SEGMENT, and these felts are opaque to the
    /// emitter — it cannot split or shift their bytes — so machine data must
    /// land on a 4-byte boundary. [`TranscriptReplay::assert_appendable`] is the
    /// loud check.
    ///
    /// This is not a real restriction for the FRI-verifier scope: every
    /// production rendering is a multiple of four bytes. A commitment root is
    /// 32, a Goldilocks felt streams as 8, a cubic-extension felt as 24.
    pub fn append_halves(&mut self, halves: &[Felt]) {
        self.assert_appendable();
        self.segment.push(SegPiece::Halves(halves.to_vec()));
        self.segment_len += BYTES_PER_HALF * halves.len();
        // Absorbing invalidates the buffer: a later challenge must depend on
        // this input, so bytes squeezed before it are dropped.
        self.out_pos = SQUEEZE_LEN;
        self.buf = None;
    }

    /// Absorb one machine word — its four lanes as four halves, 16 bytes.
    ///
    /// The word must be a `u32`-half word (a keccak state/digest word, which is
    /// where transcript-bound data comes from). Feeding one whose lanes are full
    /// felts is not a silent miscoding: the halves end up as lanes of a keccak
    /// input word, and the adapter refuses anything at or above `2^32` with
    /// `LfmExecError::NotU32Half` — and would be unprovable regardless, since the
    /// bus range-checks them.
    pub fn append_word(&mut self, b: &mut LfmBuilder, w: Cell) {
        let lanes = b.unpack(w);
        self.append_halves(&lanes);
    }

    /// Absorb a 32-byte keccak digest carried as two machine words — the shape a
    /// commitment root arrives in.
    pub fn append_digest(&mut self, b: &mut LfmBuilder, words: &[Cell; DIGEST_WORDS]) {
        for w in words {
            self.append_word(b, *w);
        }
    }

    /// Absorb one base field element the way production streams it: the
    /// canonical `u64` in BIG-endian byte order, 8 bytes.
    ///
    /// `FieldElement<GoldilocksField>::stream_bytes` is
    /// `sink(&self.canonical_u64().to_be_bytes())`, so the endianness flip is
    /// real work for this machine — see [`felt_be_halves`] for the gadget and
    /// its cost.
    pub fn append_felt(&mut self, b: &mut LfmBuilder, v: Felt) {
        let halves = felt_be_halves(b, v);
        self.append_halves(&halves);
    }

    /// Absorb one cubic-extension element: coordinates 0, 1, 2, each as its own
    /// 8 big-endian bytes — 24 bytes in total.
    ///
    /// Coordinate order is FORWARD and was verified against the source, because
    /// the file offers both orders and picking the wrong one is invisible until
    /// a challenge diverges: `FieldElement<Degree3GoldilocksExtensionField>`'s
    /// `write_bytes_be` (which `stream_bytes` calls) writes components 0, 1, 2,
    /// while the REVERSED 2, 1, 0 order belongs to the raw `[FpE; 3]` array
    /// type. Different types, no contradiction — but do not "fix" this to match
    /// the other impl.
    pub fn append_ext(&mut self, b: &mut LfmBuilder, coords: [Felt; 3]) {
        for c in coords {
            self.append_felt(b, c);
        }
    }

    /// Absorb a compile-time constant byte string of ANY length, anywhere in the
    /// segment. No alignment requirement, no builder — the bytes are stored
    /// unpacked and interned at the squeeze.
    ///
    /// ## Why packing is per SEGMENT, not per append
    ///
    /// Append boundaries are not machine-visible. Between two finalize points
    /// the production hasher sees one concatenated byte stream; `append_bytes`
    /// boundaries leave no trace in the digest input. Every length here is
    /// compile-time. So the emitter's correct unit of packing is the segment,
    /// and it packs by concatenating consecutive constant runs and only THEN
    /// chunking into halves.
    ///
    /// That is what makes "a partial half in the middle of a segment that the
    /// next append must continue into" impossible rather than merely rejected:
    /// when the emitter packs, it already holds every later constant in the
    /// segment. Appending `b"abc"` then `b"de"` yields the five-byte run
    /// `abcde`, chunked as two halves — it is not two independently packed
    /// pieces. **Do not reintroduce per-append packing.**
    ///
    /// Segment prefixes are safe by construction too: every segment after the
    /// first begins with the 32-byte reversed digest, a multiple of four.
    ///
    /// ## The one case that remains, deliberately unbuilt
    ///
    /// A constant of length ≢ 0 (mod 4) followed by MACHINE data — a 27-byte
    /// domain tag ahead of a root word, say — leaves the dynamic value straddling
    /// a half boundary, and re-aligning opaque felts by 1–3 bytes needs a
    /// byte-level splice (BitDec-32 per affected half, or a byte-table route).
    /// [`TranscriptReplay::append_halves`] rejects it loudly. It arises only in
    /// the statement-absorb leg, at a volume of a few dozen halves per proof, and
    /// never in FRI or Merkle traffic; when that leg is built it gets a
    /// `splice_misaligned(constant_prefix_len, dynamic_halves)` helper. That is
    /// an extension point, not a redesign.
    pub fn append_const_bytes(&mut self, bytes: &[u8]) {
        self.segment.push(SegPiece::Const(bytes.to_vec()));
        self.segment_len += bytes.len();
        self.out_pos = SQUEEZE_LEN;
        self.buf = None;
    }

    fn assert_appendable(&self) {
        assert_eq!(
            self.segment_len % BYTES_PER_HALF,
            0,
            "machine-computed data must start on a 4-byte boundary of the segment, \
             but {} bytes are already absorbed: a constant of length not a multiple \
             of four leaves the dynamic value straddling a half, which needs the \
             byte-level splice that only the statement-absorb leg will build",
            self.segment_len
        );
    }

    /// Absorb machine-computed data that does NOT start on a 4-byte boundary.
    ///
    /// Same bytes as [`TranscriptReplay::append_halves`], but it permits the
    /// misalignment that method rejects, and pays for it: each half then
    /// straddles two output halves and has to be split byte-wise (see
    /// [`split_half`] for the gadget and its cost). Use the aligned method
    /// wherever the encoding allows — this one exists for the statement leg,
    /// where a 30-byte domain tag and a 1-byte `fri` field between fixed-width
    /// fields make misalignment unavoidable.
    ///
    /// The splice itself happens in [`TranscriptReplay::pack_segment`], not
    /// here, because only the packer knows the byte cursor.
    pub fn append_halves_misaligned(&mut self, halves: &[Felt]) {
        self.segment.push(SegPiece::Halves(halves.to_vec()));
        self.segment_len += BYTES_PER_HALF * halves.len();
        self.out_pos = SQUEEZE_LEN;
        self.buf = None;
    }

    /// Absorb `byte_len` machine-computed bytes carried in
    /// `ceil(byte_len / 4)` halves — the general case, where the byte string's
    /// length need not be a multiple of four.
    ///
    /// The trailing half is masked to its live bytes and its unused high bytes
    /// are pinned to zero (see [`Packer::push_masked`]). That matters for any
    /// length-prefixed field: `public_output` in the epoch statement is
    /// collected one byte per COMMIT operation, so its length is whatever the
    /// workload produced and is not aligned in general.
    pub fn append_bytes_misaligned(&mut self, halves: &[Felt], byte_len: usize) {
        assert_eq!(
            halves.len(),
            byte_len.div_ceil(BYTES_PER_HALF),
            "byte_len must match the supplied halves"
        );
        let full = byte_len / BYTES_PER_HALF;
        let rem = byte_len % BYTES_PER_HALF;
        if full > 0 {
            self.segment.push(SegPiece::Halves(halves[..full].to_vec()));
        }
        if rem > 0 {
            self.segment.push(SegPiece::Partial(halves[full], rem));
        }
        self.segment_len += byte_len;
        self.out_pos = SQUEEZE_LEN;
        self.buf = None;
    }

    /// Packs the segment into `u32` halves, walking it at BYTE granularity.
    ///
    /// Constant bytes accumulate host-side; a machine half drops straight in
    /// when the cursor is 4-byte aligned — the path every aligned program takes,
    /// which must stay instruction-free — and is split when it is not. The
    /// packer is the only place that knows the cursor, which is why the splice
    /// lives here rather than at the append.
    fn pack_segment(&self, b: &mut LfmBuilder) -> Vec<Felt> {
        let mut p = Packer {
            out: Vec::new(),
            partial: Partial::Const(Vec::new()),
        };
        for piece in &self.segment {
            match piece {
                SegPiece::Const(bytes) => p.push_const(b, bytes),
                SegPiece::Halves(halves) => {
                    for h in halves {
                        p.push_half(b, *h);
                    }
                }
                SegPiece::Partial(v, nbytes) => p.push_masked(b, *v, *nbytes),
            }
        }
        p.finish(b)
    }

    /// `DefaultTranscript::sample()` — finalize, reverse the 32 digest bytes,
    /// re-absorb them, return them.
    ///
    /// The returned bytes and the re-absorbed bytes are the SAME 32 bytes; one
    /// keccak row produces both. Also invalidates the output buffer, exactly as
    /// production does.
    pub fn sample(&mut self, b: &mut LfmBuilder) -> [Cell; DIGEST_WORDS] {
        let (_plain, rev) = self.squeeze(b);
        self.buf = None;
        self.out_pos = SQUEEZE_LEN;
        rev
    }

    /// One squeeze: emits the keccak row over the current segment, sets the
    /// segment to the reversed digest, and hands back both digests — the plain
    /// one because candidates are read off it, the reversed one because it is
    /// what `sample()` returns.
    fn squeeze(&mut self, b: &mut LfmBuilder) -> ([Cell; DIGEST_WORDS], [Cell; DIGEST_WORDS]) {
        let packed = self.pack_segment(b);
        let (plain, rev) = edsl::keccak256_with_rev(b, &packed, self.segment_len);
        // The transcript absorbs the reversed bytes into a freshly reset hasher,
        // so they are the WHOLE of the next segment, not a suffix of this one.
        let mut halves = Vec::with_capacity(SQUEEZE_HALVES);
        for w in rev {
            halves.extend_from_slice(&b.unpack(w));
        }
        self.segment = vec![SegPiece::Halves(halves)];
        self.segment_len = SQUEEZE_LEN;
        (plain, rev)
    }

    /// Refill the output buffer with one squeeze, as `next_sample_u64` does.
    fn refill(&mut self, b: &mut LfmBuilder) {
        let (plain, _rev) = self.squeeze(b);
        self.buf = Some(SqueezeBuf {
            words: plain,
            lanes: [None; DIGEST_WORDS],
        });
        self.out_pos = 0;
    }

    /// The next 64-bit candidate, refilling when fewer than 8 bytes remain.
    ///
    /// Returns the candidate as its two `u32` halves rather than a felt: a
    /// candidate is a 64-BIT integer and values in `[p, 2^64)` are not
    /// felt-representable, so it cannot be one cell until it has been range-
    /// checked. Consumers either check it ([`TranscriptReplay::sample_felt`]) or
    /// use only the low half ([`TranscriptReplay::sample_u64_pow2`]).
    pub fn next_candidate(&mut self, b: &mut LfmBuilder) -> Candidate {
        if self.out_pos + CANDIDATE_BYTES > SQUEEZE_LEN {
            self.refill(b);
        }
        debug_assert_eq!(
            self.out_pos % CANDIDATE_BYTES,
            0,
            "candidates are the buffer's only consumer, so out_pos moves in 8s"
        );
        let i = self.out_pos / CANDIDATE_BYTES;
        debug_assert!(i < CANDIDATES_PER_SQUEEZE);
        // Candidate i is the plain digest's u64 lane 3 − i (the reversal
        // cancellation), and lane j is halves 2j (low) and 2j + 1 (high).
        let lo = self.half(b, 2 * (CANDIDATES_PER_SQUEEZE - 1 - i));
        let hi = self.half(b, 2 * (CANDIDATES_PER_SQUEEZE - 1 - i) + 1);
        self.out_pos += CANDIDATE_BYTES;
        Candidate { lo, hi }
    }

    /// Half `h` of the buffered digest: lane `h % 4` of word `h / 4`, unpacking
    /// that word on first use.
    fn half(&mut self, b: &mut LfmBuilder, h: usize) -> Felt {
        let buf = self
            .buf
            .as_mut()
            .expect("next_candidate refills before reading");
        let (w, l) = (h / 4, h % 4);
        let lanes = match buf.lanes[w] {
            Some(lanes) => lanes,
            None => {
                let lanes = b.unpack(buf.words[w]);
                buf.lanes[w] = Some(lanes);
                lanes
            }
        };
        lanes[l]
    }

    /// One base-field challenge: `GoldilocksField::sample_field_element_from`
    /// with the rejection branch replaced by a constraint (see the module docs).
    pub fn sample_felt(&mut self, b: &mut LfmBuilder) -> Felt {
        let c = self.next_candidate(b);
        assert_canonical(b, c);
        candidate_to_felt(b, c)
    }

    /// One cubic-extension challenge: three independent base draws in
    /// coordinate order 0, 1, 2 — which is what
    /// `Degree3GoldilocksExtensionField::sample_field_element_from` does
    /// (`core::array::from_fn` evaluates in index order).
    ///
    /// This is the production shape: the STARK verifier's challenges are
    /// extension elements, so an ext draw is where the completeness bound is
    /// paid three times over.
    pub fn sample_ext(&mut self, b: &mut LfmBuilder) -> Ext {
        let a0 = self.sample_felt(b);
        let a1 = self.sample_felt(b);
        let a2 = self.sample_felt(b);
        b.pack_ext(a0, a1, a2)
    }

    /// `sample_u64(1 << nbits)` — the low `nbits` bits of one candidate, as bits
    /// low-to-high.
    ///
    /// No canonicity guard and no rejection, because production has none here:
    /// `threshold = upper_bound.wrapping_neg() % upper_bound` is 0 at every
    /// power of two, so the loop in `sample_u64` accepts its first candidate
    /// unconditionally and returns `candidate % 2^nbits`. An out-of-range
    /// candidate is perfectly legal for this draw — which is why `u64` draws
    /// contribute NOTHING to the completeness bound.
    ///
    /// `nbits ≤ 32` keeps the answer inside the candidate's low half. The bound
    /// is real rather than defensive: FRI query indices are bounded by the LDE
    /// domain, which is ≤ 2^25 here.
    pub fn sample_u64_pow2(&mut self, b: &mut LfmBuilder, nbits: usize) -> Vec<Bit> {
        assert!(
            (1..=32).contains(&nbits),
            "sample_u64_pow2: nbits must be in 1..=32, got {nbits} — above 32 the \
             answer would span both halves of the candidate"
        );
        let c = self.next_candidate(b);
        b.bit_dec(c.lo, nbits)
    }

    /// Emit-time buffer position, for tests that pin the consumption schedule.
    pub fn out_pos(&self) -> usize {
        self.out_pos
    }

    /// Emit-time segment length in bytes, for the same reason.
    pub fn segment_len(&self) -> usize {
        self.segment_len
    }
}

/// Constrains a candidate to be a canonical field element, i.e. `< p`.
///
/// `p = 2^64 − 2^32 + 1`, so `p − 1 = (2^32 − 1)·2^32` and
/// `p = (2^32 − 1)·2^32 + 1`. For `candidate = hi·2^32 + lo` with both halves
/// below `2^32`:
///
/// - `hi < 2^32 − 1` ⇒ `candidate ≤ (2^32 − 2)·2^32 + (2^32 − 1)`
///   `= (2^32 − 1)·2^32 − 1 < p`, always in range;
/// - `hi = 2^32 − 1` ⇒ `candidate = (p − 1) + lo`, in range iff `lo = 0`.
///
/// So `candidate ≥ p ⟺ hi = 2^32 − 1 ∧ lo ≠ 0`. (The `LFM_BITDEC` chip proves
/// canonicity of a 64-bit decomposition with the same predicate over the top and
/// bottom 32 bits — see `chips::bitdec`.)
///
/// The guard is then a single division. `g = (2^32 − 1) − hi` is zero exactly
/// when `hi = 2^32 − 1`, and `LFM_BALU` constrains division as
/// `SEL_DIV·(B·OUT − A) = 0`: with `B = 0` that reads `−A = 0`, forcing `A = 0`
/// and leaving `OUT` free. So `div(lo, g)` is provable iff `g ≠ 0 ∨ lo = 0` —
/// the exact negation of the reject condition, in one instruction with nothing
/// hinted and nothing to verify. It is the same assert-via-division mechanism
/// `LfmBuilder::assert_eq` is built from.
///
/// Both halves must be canonical `u32`s for the derivation to hold. They are:
/// they come from `Unpack` of a `LFM_KECCAK` output word, whose halves the
/// keccak adapter range-checks (`keccak_rejects_non_u32_half`).
pub fn assert_canonical(b: &mut LfmBuilder, c: Candidate) {
    let hi_max = b.felt_const(FE::from(HI_MAX));
    let g = b.sub(hi_max, c.hi);
    let _ = b.div(c.lo, g);
}

/// `hi·2^32 + lo` as a field element.
///
/// Only equal to the candidate's INTEGER value once [`assert_canonical`] has
/// pinned that value below `p`; without the guard this silently wraps (a
/// candidate of `p` becomes `0`).
pub fn candidate_to_felt(b: &mut LfmBuilder, c: Candidate) -> Felt {
    let two32 = b.felt_const(FE::from(1u64 << 32));
    b.mul_add(c.hi, two32, c.lo)
}

// =============================== the packer ===============================

/// The half currently under construction.
enum Partial {
    /// Its bytes so far, all compile-time. Always fewer than four.
    Const(Vec<u8>),
    /// A machine value occupying the LOW `filled` bytes of the half, with
    /// `filled` in `1..4`. Its unfilled high bytes are zero, so completing it is
    /// an addition rather than an or.
    Mixed(Felt, usize),
}

/// Emits a segment's `u32` halves from a byte-granular walk of its pieces.
struct Packer {
    out: Vec<Felt>,
    partial: Partial,
}

/// The little-endian value of up to four bytes.
fn le_value(bytes: &[u8]) -> u64 {
    bytes
        .iter()
        .enumerate()
        .fold(0u64, |acc, (i, &v)| acc | (u64::from(v) << (8 * i)))
}

impl Packer {
    fn filled(&self) -> usize {
        match &self.partial {
            Partial::Const(v) => v.len(),
            Partial::Mixed(_, f) => *f,
        }
    }

    fn push_const(&mut self, b: &mut LfmBuilder, bytes: &[u8]) {
        for &byte in bytes {
            match core::mem::replace(&mut self.partial, Partial::Const(Vec::new())) {
                Partial::Const(mut v) => {
                    v.push(byte);
                    if v.len() == BYTES_PER_HALF {
                        let c = b.felt_const(FE::from(le_value(&v)));
                        self.out.push(c);
                        v.clear();
                    }
                    self.partial = Partial::Const(v);
                }
                Partial::Mixed(m, filled) => {
                    // The byte lands above what is already there, and the high
                    // bytes are zero, so `add` is exactly an or.
                    let w = b.felt_const(FE::from(u64::from(byte) << (8 * filled)));
                    let m = b.add(m, w);
                    if filled + 1 == BYTES_PER_HALF {
                        self.out.push(m);
                        self.partial = Partial::Const(Vec::new());
                    } else {
                        self.partial = Partial::Mixed(m, filled + 1);
                    }
                }
            }
        }
    }

    /// Merges `v` into the half under construction at byte offset `filled`.
    /// The destination's higher bytes are zero, so `mul_add` is exactly an or.
    fn merge(&mut self, b: &mut LfmBuilder, v: Felt, filled: usize) -> Felt {
        let shift = b.felt_const(FE::from(1u64 << (8 * filled)));
        match core::mem::replace(&mut self.partial, Partial::Const(Vec::new())) {
            Partial::Const(c) => {
                let base = b.felt_const(FE::from(le_value(&c)));
                b.mul_add(v, shift, base)
            }
            Partial::Mixed(m, _) => b.mul_add(v, shift, m),
        }
    }

    /// Places a machine value carrying `nbytes` live bytes at the cursor.
    ///
    /// One routine covers both a whole half (`nbytes == 4`) and the masked tail
    /// of an odd-length byte string, because they differ only in width.
    fn push_partial(&mut self, b: &mut LfmBuilder, v: Felt, nbytes: usize) {
        debug_assert!((1..=BYTES_PER_HALF).contains(&nbytes));
        let filled = self.filled();
        if filled == 0 {
            // Nothing to merge with: `v` already sits in the low bytes of a
            // fresh half and its high bytes are zero. No instructions — the path
            // every aligned program takes, which must stay free or every
            // registered digest moves.
            if nbytes == BYTES_PER_HALF {
                self.out.push(v);
            } else {
                self.partial = Partial::Mixed(v, nbytes);
            }
            return;
        }
        let room = BYTES_PER_HALF - filled;
        if nbytes < room {
            let merged = self.merge(b, v, filled);
            self.partial = Partial::Mixed(merged, filled + nbytes);
        } else if nbytes == room {
            let merged = self.merge(b, v, filled);
            self.out.push(merged);
            self.partial = Partial::Const(Vec::new());
        } else {
            // Crosses the boundary: the low `room` bytes finish this half and
            // the rest opens the next.
            let (lo, hi) = split_half(b, v, room);
            let merged = self.merge(b, lo, filled);
            self.out.push(merged);
            self.partial = Partial::Mixed(hi, nbytes - room);
        }
    }

    fn push_half(&mut self, b: &mut LfmBuilder, d: Felt) {
        self.push_partial(b, d, BYTES_PER_HALF);
    }

    /// Masks a trailing half to its `nbytes` live bytes and PINS the rest to
    /// zero, then places it.
    ///
    /// The zero-pin is a soundness obligation, not tidiness: the high bytes of
    /// an arena-supplied felt are otherwise unconstrained, and without it a
    /// prover could put arbitrary content there. Those bytes are past the
    /// encoding's length prefix, so they would change the absorbed byte string
    /// while the length said otherwise.
    fn push_masked(&mut self, b: &mut LfmBuilder, v: Felt, nbytes: usize) {
        let (lo, hi) = split_half(b, v, nbytes);
        let zero = b.felt_const(FE::zero());
        b.assert_eq(hi, zero);
        self.push_partial(b, lo, nbytes);
    }

    fn finish(mut self, b: &mut LfmBuilder) -> Vec<Felt> {
        match self.partial {
            // A trailing partial half's unused high bytes are zero either way,
            // which is the property `edsl::keccak256` needs to merge the padding
            // constant with an `add`.
            Partial::Const(v) if v.is_empty() => {}
            Partial::Const(v) => {
                let c = b.felt_const(FE::from(le_value(&v)));
                self.out.push(c);
            }
            Partial::Mixed(m, _) => self.out.push(m),
        }
        self.out
    }
}

/// Splits a `u32` half into its low `k` bytes and its high `4 − k` bytes.
///
/// This is the byte-level splice the misaligned statement encoding needs. A byte
/// split is not field arithmetic, so it goes through the canonical bit
/// decomposition and two weighted sums over disjoint bit ranges.
///
/// The recomposition assert is load-bearing, not a belt: `bit_dec` bounds its
/// input by `p`, not by `2^32`, and a "half" at or above `2^32` has no four-byte
/// rendering at all. Pinning `d = lo + hi·2^(8k)` forces `d < 2^32` and the
/// split's correctness in the same constraint.
///
/// Cost: one `LFM_BITDEC` row and ~33 `LFM_BALU` rows per spliced half. It only
/// ever runs on the statement leg — a few dozen halves per proof — and never in
/// FRI or Merkle traffic.
pub fn split_half(b: &mut LfmBuilder, d: Felt, k: usize) -> (Felt, Felt) {
    assert!(
        (1..BYTES_PER_HALF).contains(&k),
        "split_half: k must be in 1..4, got {k}"
    );
    let bits = b.bit_dec(d, 8 * BYTES_PER_HALF);
    let lo = edsl::bits_to_felt(b, &bits[..8 * k]);
    let hi = edsl::bits_to_felt(b, &bits[8 * k..]);
    let shift = b.felt_const(FE::from(1u64 << (8 * k)));
    let recomposed = b.mul_add(hi, shift, lo);
    b.assert_eq(d, recomposed);
    (lo, hi)
}

/// The two `u32` halves of a base felt's 8-byte BIG-endian rendering — what
/// `append_field_element` puts on the wire, expressed in the machine's
/// little-endian half convention.
///
/// ## The derivation
///
/// Write `v = hi·2^32 + lo`. Big-endian, `v`'s bytes are `hi`'s four bytes
/// most-significant-first, then `lo`'s. Half `h` of the segment is the LE `u32`
/// of segment bytes `4h..4h+4`, so
///
/// - half 0 = `byteswap32(hi)` — the HIGH word leads in big-endian order,
/// - half 1 = `byteswap32(lo)`.
///
/// A byte swap is not field arithmetic, so it goes through the canonical bit
/// decomposition: bit `j` of byte `k` must land at bit `j` of byte `3 − k`,
/// which is just a different constant weight per bit. Each half is therefore one
/// 32-term linear form, and the whole byte permutation lives in the weights
/// rather than in any emitted instruction.
///
/// ## Cost
///
/// One `LFM_BITDEC` row plus 64 `LFM_BALU` rows (per half: a `Mul` to open the
/// accumulator, then 31 `MulAdd`s), and the 32 weight constants are interned
/// once and shared by both halves — they are the powers `2^0..2^31`, since
/// `j + 8(3 − k)` runs over `0..32` bijectively.
///
/// `bit_dec` also enforces canonicity (`< p`), which is exactly right: production
/// renders `canonical_u64()`.
///
/// Note for callers re-absorbing a value the transcript just produced: a
/// challenge from [`TranscriptReplay::sample_felt`] arrives as a recomposed
/// `Felt` and is decomposed again here. That round trip is one redundant
/// `BitDec`; carrying the halves through would avoid it, and is worth doing only
/// if a profile says so.
pub fn felt_be_halves(b: &mut LfmBuilder, v: Felt) -> [Felt; 2] {
    let bits = b.bit_dec(v, 64);
    core::array::from_fn(|h| {
        // Half 0 carries the value's HIGH 32 bits: they lead in big-endian order.
        let first = if h == 0 { 32 } else { 0 };
        let mut acc: Option<Felt> = None;
        for k in 0..4 {
            for j in 0..8 {
                let weight = b.felt_const(FE::from(1u64 << (j + 8 * (3 - k))));
                let bit = bits[first + 8 * k + j].as_felt();
                acc = Some(match acc {
                    None => b.mul(bit, weight),
                    Some(a) => b.mul_add(bit, weight, a),
                });
            }
        }
        acc.expect("32 bits per half")
    })
}

/// Per-candidate probability that the production sampler rejects: there are
/// `2^64 − p = 2^32 − 1` out-of-range values among the `2^64` a candidate can
/// take.
///
/// Only `sample_field_element` draws are exposed — `sample_u64` at a power-of-two
/// bound never rejects.
pub fn reject_probability_per_candidate() -> f64 {
    ((1u64 << 32) - 1) as f64 / 2f64.powi(64)
}

/// Upper bound on the probability that a transcript with `base_draws` base-field
/// challenge draws rejects at least once — i.e. that the emitted zero-rejection
/// program cannot prove it.
///
/// A cubic-extension challenge is THREE base draws, so pass `3 · ext_draws`.
/// The union bound is what makes this an upper bound; the exact value is
/// `1 − (1 − q)^n`, indistinguishable at these magnitudes.
pub fn reject_probability_per_proof(base_draws: usize) -> f64 {
    base_draws as f64 * reject_probability_per_candidate()
}
