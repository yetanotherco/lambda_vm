"""L6 — the lincomb2 joint-schedule counting argument, checked against the chips.

Model transcribed by READING `prover/src/tables/ecdas2.rs` (288 constraints, the
schedule-relevant ones are idx 0..=27) and `ecsm2.rs` (693 constraints; the bus
wiring at :543-923). Convolution carries are out of scope here — they are the
width audit's subject (`../lincomb2/WIDTH-AUDIT.md`) — this file is about the
SCHEDULE: which rows exist, in what order, and which digits they consume.

Sections:

  L6-A  per-row schedule constraints are consistent, and force the honest shape
        on live rows (round monotonicity, the add/addend agreement table).
  L6-B  the 2x JointBit multiplicity is strictly stronger than 1x — the 1x
        variant admits a WRONG ADDEND at a round where both digits are set.
  L6-C  *** THE BREAK, NOW CLOSED ***  padding rows (MU = 0) could send live
        JointBit digits, because the send's multiplicity is `Column(D1)` and no
        constraint tied D1/D2 to MU. Two phantom rows satisfy a round's 2x
        count with NO add row on the chain — a set bit of u1 is "consumed"
        without ever being added. Reproduced here as the ABLATION of a defence
        the chip now carries; L6-E is what it was worth.
  L6-D  the fix — `(1−MU)·{D1, D2, S1, S2, S3, S_CORR} = 0`, mirroring
        `ecdas.rs` idx 4 `NEXT_OP·(1−MU) = 0` — turns L6-C UNSAT. It LANDED as
        `ecdas2.rs:988-1003`, idx 22..=27; `gate2_common.chip_state()` parses
        the emitted expression and cross-checks the gated column set against the
        columns that actually supply a bus multiplicity.

Note this file keeps its own `Row` class rather than using
`gate2_common.Ecdas2Row`: it is the historical derivation of the break, and it
must be able to model the pre-fix chip.

Run:  <venv>/bin/python l6_joint_counting.py
"""

import sys

from z3 import And, Bool, Distinct, If, Implies, Int, Not, Or, Solver, Sum, sat, unsat


def bit(s, name):
    v = Int(name)
    s.add(Or(v == 0, v == 1))
    return v


class Row:
    """One ECDAS2 row's schedule columns."""

    def __init__(self, s, tag, fix_padding_gate=False):
        self.mu = bit(s, f"MU_{tag}")
        self.op = bit(s, f"OP_{tag}")
        self.nb = bit(s, f"NB_{tag}")
        self.d1 = bit(s, f"D1_{tag}")
        self.d2 = bit(s, f"D2_{tag}")
        self.s1 = bit(s, f"S1_{tag}")
        self.s2 = bit(s, f"S2_{tag}")
        self.s3 = bit(s, f"S3_{tag}")
        self.sc = bit(s, f"SC_{tag}")
        self.ph1 = bit(s, f"PH1_{tag}")
        self.ph2 = bit(s, f"PH2_{tag}")
        self.round = Int(f"ROUND_{tag}")
        s.add(self.round >= 0, self.round <= 255)  # AreBytes(ROUND), MU-gated

        # --- ecdas2.rs constraint indices 11..=21, transcribed ---
        s.add(self.ph1 * self.ph2 == 0)                                    # 11
        s.add(self.op * self.nb == 0)                                      # 12
        s.add((1 - self.op) * (self.nb - self.d1 - self.d2
                               + self.d1 * self.d2) == 0)                  # 13
        s.add(self.op - self.s1 - self.s2 - self.s3 - self.sc == 0)        # 14
        s.add((1 - self.ph1) * self.d1 == 0)                               # 15
        s.add((1 - self.ph1) * self.d2 == 0)                               # 16
        s.add(self.ph1 * self.sc == 0)                                     # 17
        s.add(self.ph1 * (self.s1 + self.s3 - self.op * self.d1) == 0)     # 18
        s.add(self.ph1 * (self.s2 + self.s3 - self.op * self.d2) == 0)     # 19
        s.add(self.mu * (1 - self.ph1 - self.ph2) * (self.s2 - 1) == 0)    # 20
        s.add(self.ph2 * (self.sc - 1) == 0)                               # 21

        if fix_padding_gate:
            # LANDED as ecdas2.rs idx 22, 23, mirroring ecdas.rs idx 4
            # (`NEXT_OP·(1−MU) = 0`). The chip also gates the four selectors
            # (idx 24..=27); only the digit half matters to this file.
            s.add((1 - self.mu) * self.d1 == 0)
            s.add((1 - self.mu) * self.d2 == 0)

    def sends_stream(self, stream):
        """Multiplicity of this row's JointBit send. NOT MU-gated in the chip:
        `Multiplicity::Column(cols::D1)` (ecdas2.rs:601-612). What makes a
        padding row inert is idx 22, 23, not the send."""
        return self.d1 if stream == 1 else self.d2


def hdr(t):
    print()
    print("=" * 74)
    print(t)
    print("=" * 74)


# ── L6-A: the per-row add/addend agreement table ────────────────────────────

def l6_a():
    hdr("L6-A — per-row schedule constraints (live main-chain rows)")
    results = []

    # (a) On a live main-chain ADD, the digits select exactly one addend, and
    #     the (0,0) digit pair is unsatisfiable (no spurious add).
    for d1v, d2v, want in [(1, 0, "S1"), (0, 1, "S2"), (1, 1, "S3"), (0, 0, None)]:
        s = Solver()
        r = Row(s, "add")
        s.add(r.mu == 1, r.ph1 == 1, r.op == 1, r.d1 == d1v, r.d2 == d2v)
        if want is None:
            verdict = s.check()
            ok = verdict == unsat
            print(f"   digits (0,0) on an add            : {verdict} "
                  f"({'no spurious add possible' if ok else 'SPURIOUS ADD'})")
            results.append(ok)
        else:
            # the intended selector must be forced: negating it is UNSAT
            s.push()
            s.add(Not(getattr(r, want.lower()) == 1))
            verdict = s.check()
            s.pop()
            ok = verdict == unsat
            print(f"   digits ({d1v},{d2v}) force {want:2}          : "
                  f"{verdict} ({'forced' if ok else 'NOT FORCED'})")
            results.append(ok)

    # (b) On a live main-chain DOUBLE, no addend is consumed.
    s = Solver()
    r = Row(s, "dbl")
    s.add(r.mu == 1, r.ph1 == 1, r.op == 0)
    s.add(r.s1 + r.s2 + r.s3 + r.sc > 0)
    verdict = s.check()
    ok = verdict == unsat
    print(f"   doubling consumes no addend       : {verdict} "
          f"({'forced' if ok else 'ADDEND LEAK'})")
    results.append(ok)

    # (c) NB = D1 or D2 on a doubling: a set digit forces an add to follow.
    s = Solver()
    r = Row(s, "nb")
    s.add(r.mu == 1, r.ph1 == 1, r.op == 0, Or(r.d1 == 1, r.d2 == 1), r.nb == 0)
    verdict = s.check()
    ok = verdict == unsat
    print(f"   set digit forces NB = 1           : {verdict} "
          f"({'forced' if ok else 'ADD CAN BE DROPPED'})")
    results.append(ok)

    # (d) Round monotonicity: successor round is round − 1 + NB, and an add
    #     (OP = 1) always has NB = 0, so two consecutive steps strictly decrease.
    s = Solver()
    r = Row(s, "mono")
    s.add(r.mu == 1, r.op == 1, r.nb == 1)
    verdict = s.check()
    ok = verdict == unsat
    print(f"   add-after-add impossible (OP·NB)  : {verdict} "
          f"({'forced' if ok else 'ROUND CAN STALL'})")
    results.append(ok)

    return all(results)


# ── L6-B: is the 2x JointBit multiplicity really stronger than 1x? ──────────

def l6_b():
    hdr("L6-B — the 2x JointBit multiplicity vs 1x")
    print("   Scenario: round r with u1_bit(r) = u2_bit(r) = 1. The honest chain")
    print("   has a double and an add, both carrying (D1,D2) = (1,1), so the add")
    print("   selects S3 = P12. Can a prover make the add select P2 instead?")
    print()

    def scenario(mult):
        s = Solver()
        dbl = Row(s, "d")
        add = Row(s, "a")
        for r in (dbl, add):
            s.add(r.mu == 1, r.ph1 == 1)
        s.add(dbl.op == 0, add.op == 1)
        s.add(dbl.round == add.round)          # the add shares the double's round
        s.add(dbl.nb == 1)                     # the add exists
        # JointBit balance at this round, both streams, u1_bit = u2_bit = 1
        s.add(dbl.sends_stream(1) + add.sends_stream(1) == mult)
        s.add(dbl.sends_stream(2) + add.sends_stream(2) == mult)
        # the attack: the add consumes something other than P12
        s.add(add.s3 == 0)
        return s.check()

    v1 = scenario(1)
    v2 = scenario(2)
    print(f"   multiplicity 1x : {v1}   "
          f"{'<-- WRONG ADDEND REACHABLE' if v1 == sat else ''}")
    print(f"   multiplicity 2x : {v2}   "
          f"{'(blocked)' if v2 == unsat else '<-- STILL REACHABLE'}")
    print()
    print("   Reading: with 1x the prover splits the two digits across the two")
    print("   rows (double takes D1, add takes D2), the counts still balance, and")
    print("   the add adds P2 where the schedule calls for P12 — a wrong Q.")
    print("   2x forces BOTH rows to carry BOTH digits, which pins S3. The 2x")
    print("   claim is CONFIRMED: 2 = 1+1 is the only decomposition because each")
    print("   row's multiplicity is a single IS_BIT column.")
    return v1 == sat and v2 == unsat


# ── L6-C: the break ─────────────────────────────────────────────────────────

def l6_c(fix=False):
    hdr(f"L6-{'D' if fix else 'C'} — padding rows as phantom digit senders"
        f"{' (WITH the proposed fix)' if fix else ''}")

    # Step 1: is a padding row with a live digit send even satisfiable?
    s = Solver()
    ph = Row(s, "pad", fix_padding_gate=fix)
    s.add(ph.mu == 0, ph.d1 == 1)
    v_row = s.check()
    print(f"   a MU=0 row with D1=1 satisfies all row constraints : {v_row}")

    # Step 2: the schedule-level consequence. Round r, u1_bit(r) = 1, and the
    # prover wants NO add at round r (so u1's bit is never added to the chain).
    s = Solver()
    dbl = Row(s, "d", fix_padding_gate=fix)
    s.add(dbl.mu == 1, dbl.ph1 == 1, dbl.op == 0)
    s.add(dbl.nb == 0)                       # <-- no add follows at this round
    phantoms = [Row(s, f"p{i}", fix_padding_gate=fix) for i in range(4)]
    for p in phantoms:
        s.add(p.mu == 0)                     # padding rows
        s.add(p.round == dbl.round)          # aimed at the victim round
    # JointBit balance for stream 1 at this round must equal 2·u1_bit(r) = 2.
    total = dbl.sends_stream(1) + Sum([p.sends_stream(1) for p in phantoms])
    s.add(total == 2)
    v_sched = s.check()

    print(f"   round r: u1_bit(r)=1 balanced with NO add on chain  : {v_sched}")
    if v_sched == sat:
        m = s.model()
        live = [i for i, p in enumerate(phantoms) if m.evaluate(p.d1).as_long() == 1]
        print(f"      witness: double has D1={m.evaluate(dbl.d1)}, NB=0 (no add), "
              f"phantom rows {live} each send D1=1 at the same ROUND")
    print()
    if not fix:
        print("   *** BREAK (ablation) *** The JointBit send's multiplicity is")
        print("   `Multiplicity::Column(cols::D1)` (ecdas2.rs:601-612) and, before")
        print("   idx 22..=27, NOTHING tied D1/D2 to MU. A padding row is otherwise")
        print("   inert (its Ecdas, AreBytes and IsHalfword interactions are all")
        print("   MU-gated, its Addend receive is ΣS-gated) but its digit send was")
        print("   LIVE. Two phantom rows per targeted round satisfy the 2x count")
        print("   while the chain skips the add entirely.")
    else:
        print("   With `(1−MU)·{D1, D2, S1, S2, S3, S_CORR} = 0` — the exact shape")
        print("   of `ecdas.rs` idx 4 `NEXT_OP·(1−MU) = 0` — both queries go UNSAT.")
        print("   This is `ecdas2.rs:988-1003`, idx 22..=27, in the chip today.")

    if fix:
        return v_row == unsat and v_sched == unsat
    return v_row == sat and v_sched == sat


# ── L6-E: what the break is worth — an ARBITRARY chosen recovered key ───────

def l6_e():
    hdr("L6-E — exploitability: steering the recovered key to a chosen target")
    sys.path.insert(0, "../oracle")
    from ec_ref import GX, GY, N, P, pt_add, pt_double, recover_even_y, scalar_mul
    import lincomb2_ref

    T0, _ = lincomb2_ref.t0_ref()
    G = (GX, GY)

    def chain(u1_eff, u2_eff, p1, p2, length):
        """The chain as it executes when the adds for the dropped bits are
        absent: the effective multipliers are u1_eff / u2_eff."""
        p12 = pt_add(p1, p2)
        acc = T0
        for r in range(length - 1, -1, -1):
            acc = pt_double(acc)
            e1, e2 = (u1_eff >> r) & 1, (u2_eff >> r) & 1
            if not (e1 or e2):
                continue
            addend = p12 if (e1 and e2) else (p1 if e1 else p2)
            acc = pt_add(acc, addend)
        tpow = T0
        for _ in range(length):
            tpow = pt_double(tpow)
        return pt_add(acc, (tpow[0], (P - tpow[1]) % P))

    # The attacker names the public key it wants `ecrecover` to return.
    victim_sk = 0xC0FFEE1234567890ABCDEF0011223344556677889900AABBCCDDEEFF01020304
    target = scalar_mul(victim_sk, G)
    print(f"   target pubkey (a key the attacker does NOT hold):")
    print(f"      ({target[0]:#x},")
    print(f"       {target[1]:#x})")

    # Choose effective multipliers, then solve for R. u1' = u2' = 1 gives
    # R = target − G, which is a plain point subtraction — no discrete log.
    u1_eff, u2_eff = 1, 1
    neg_g = (GX, (P - GY) % P)
    R = pt_add(target, neg_g)                     # R = target − G

    # Now inflate u1 with one extra set bit that the chain will be made to skip.
    m = 8                                          # any bit unset in u1' and u2'
    u1 = u1_eff | (1 << m)
    u2 = u2_eff
    length = max(u1.bit_length(), u2.bit_length())

    forged = chain(u1_eff, u2_eff, G, R, length)
    honest = lincomb2_ref.lincomb2(u1, G, u2, R)
    print()
    print(f"   u1 = {u1} (bit {m} will be dropped), u2 = {u2}, len = {length}")
    print(f"   honest chain  Q  = ({honest[0]:#x}, ...)")
    print(f"   forged chain  Q' = ({forged[0]:#x}, ...)")
    print(f"   Q' == target     : {forged == target}")
    print(f"   Q' != honest Q   : {forged != honest}")

    # Package it as a real ecrecover input: r = x(R), and z, s back-solved.
    r_sig = R[0]
    ok_pkg = r_sig < N and r_sig != 0
    if ok_pkg:
        v = R[1] & 1
        rinv = pow(r_sig, N - 2, N)
        z = (-u1 * r_sig) % N
        s_sig = (u2 * r_sig) % N
        gu1, gu2 = (-(rinv * z)) % N, (rinv * s_sig) % N
        ye = recover_even_y(r_sig)
        lifted = (r_sig, ye if v == 0 else (P - ye) % P)
        print()
        print(f"   ecrecover packaging: z={z:#x}")
        print(f"                        r={r_sig:#x}")
        print(f"                        s={s_sig:#x}  v={v}")
        print(f"      guest recomputes u1: {gu1 == u1}   u2: {gu2 == u2}   "
              f"lifts R: {lifted == R}")
    print()
    print("   Reading: u1' and u2' are free, so R = target − u1'·G is a plain")
    print("   point subtraction — NO discrete log is needed. The attacker gets an")
    print("   ARBITRARY chosen recovered public key, i.e. an arbitrary chosen")
    print("   transaction sender. This is strictly worse than the NUMS finding,")
    print("   which only yielded a one-parameter family.")
    return forged == target and forged != honest and ok_pkg


def main():
    a = l6_a()
    b = l6_b()
    c = l6_c(fix=False)
    d = l6_c(fix=True)
    e = l6_e()

    from gate2_common import chip_state
    st = chip_state()
    pad = st["padding_gate_detail"]
    hdr("SUMMARY")
    print(f"   chip state: (1−MU)·X gate present = {st['padding_digit_gate']}, "
          f"D_INV present = {st['dinv_relation']}")
    print(f"   gated columns {sorted(pad['gated'])}")
    print(f"   == raw bus multiplicities {sorted(pad['raw_multiplicity'])}: "
          f"{pad['exact']}")
    if pad["ungated_multiplicities"]:
        print("   *** A BUS MULTIPLICITY HAS ESCAPED THE GATE: "
              f"{sorted(pad['ungated_multiplicities'])} ***")
    print()
    print(f"   L6-A per-row schedule forcing      : {'PASS' if a else 'FAIL'}")
    print(f"   L6-B 2x multiplicity load-bearing  : {'CONFIRMED' if b else 'NOT CONFIRMED'}")
    print(f"   L6-C padding-row digit forgery     : {'REPRODUCED' if c else 'not reproduced'}")
    print(f"   L6-D proposed fix closes it        : {'YES' if d else 'NO'}")
    print(f"   L6-E arbitrary chosen sender       : {'DEMONSTRATED' if e else 'not demonstrated'}")
    print()
    if c and not st["padding_digit_gate"]:
        print("   VERDICT: L6 does NOT hold — the (1−MU)·D gate is absent from the")
        print("   chip, so L6-C is the CURRENT STATE, not an ablation.")
    elif c:
        print("   VERDICT: L6 HOLDS. L6-C is now a genuine ablation of a check the")
        print("   chip carries (idx 22..=27); with it present the forgery is UNSAT")
        print("   (L6-D). The break this file was written to demonstrate is closed.")
    else:
        print("   VERDICT: inconclusive — L6-C did not reproduce.")
    return 0 if (a and b and c and d and e) else 1


if __name__ == "__main__":
    sys.exit(main())
