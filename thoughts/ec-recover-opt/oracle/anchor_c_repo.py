"""Anchor C: differential of the repo's `crypto/ecsm` crate against the
oracle, via the repo-harness line protocol.

Checks:
  1. scalar_mul_x on 200+ random and edge (k, x) pairs
  2. exact error-path agreement (kind AND check order) on invalid inputs
  3. recover_y_canonical: even-y choice, non-residue rejection, x >= p rejection
  4. replay_double_and_add: schedule == documented MSB-first double-and-add
     derived independently from bits of k; every step's (a, lambda, r) matches
     the oracle's own chord/tangent replay; final == oracle x(k*P); k=1 echo.
"""

import random
import subprocess

from ec_ref import (EcError, GX, GY, N, P, expected_schedule, recover_even_y,
                    replay_schedule, scalar_mul, x_only_mul_ints)

HARNESS = "repo-harness/target/release/ecsm-oracle-harness"
rng = random.Random(97)


def random_valid_x():
    while True:
        x = rng.randrange(P)
        if recover_even_y(x) is not None:
            return x


def nonresidue_x():
    while True:
        x = rng.randrange(P)
        if recover_even_y(x) is None:
            return x


def run(lines):
    r = subprocess.run([HARNESS], input="\n".join(lines) + "\n",
                       capture_output=True, text=True, check=True)
    return r.stdout.splitlines()


def main():
    fails = 0

    def check(cond, msg):
        nonlocal fails
        if not cond:
            fails += 1
            print(f"FAIL {msg}")

    # ── 1. mul differential ──────────────────────────────────────────────
    cases = []
    for _ in range(180):
        cases.append((random_valid_x(), rng.randrange(1, N)))
    edge_ks = [1, 2, 3, N - 1, N - 2] + [2**i for i in (1, 8, 64, 128, 255)] \
        + [2**i - 1 for i in (2, 8, 64, 128, 256) if 2**i - 1 < N] \
        + [int("10" * 128, 2) % N, int("01" * 128, 2)]
    for k in edge_ks:
        cases.append((GX, k))
        cases.append((random_valid_x(), k))
    out = run([f"mul {x:x} {k:x}" for (x, k) in cases])
    for (x, k), line in zip(cases, out):
        want = x_only_mul_ints(x, k)
        check(line == f"ok {want:x}", f"mul k={k:x} x={x:x}: repo said {line!r}, oracle {want:x}")
    n_mul = len(cases)

    # ── 2. error paths: exact kind + check order ─────────────────────────
    nr = nonresidue_x()
    vx = random_valid_x()
    err_cases = [
        (vx, 0), (vx, N), (vx, N + 1), (vx, 2**256 - 1),
        (P, 5), (P + 1, 5), (2**256 - 1, 5),
        (nr, 5), (nonresidue_x(), rng.randrange(1, N)),
        # combined-invalid: verifies the check ORDER is scalar -> coord -> curve
        (P, 0), (P, N), (nr, 0), (nr, N + 7), (P + 3, 2**256 - 1),
    ]
    out = run([f"mul {x:x} {k:x}" for (x, k) in err_cases])
    for (x, k), line in zip(err_cases, out):
        try:
            x_only_mul_ints(x, k)
            want = "ok"
        except EcError as e:
            want = f"err {e.kind}"
        check(line == want, f"errpath k={k:x} x={x:x}: repo {line!r}, oracle {want!r}")

    # ── 3. recover_y_canonical ───────────────────────────────────────────
    ys = [random_valid_x() for _ in range(40)] + [nonresidue_x() for _ in range(20)] \
        + [P, P + 1, 2**256 - 1, 0, GX]
    out = run([f"recovery {x:x}" for x in ys])
    for x, line in zip(ys, out):
        y = recover_even_y(x)
        want = f"y {y:x}" if y is not None else "none"
        check(line == want, f"recovery x={x:x}: repo {line!r}, oracle {want!r}")
        if y is not None:
            check(y % 2 == 0, f"oracle even-y invariant broke at x={x:x}")

    # ── 4. replay: schedule + per-step transition + final ────────────────
    replay_ks = [1, 2, 3, 5, 7, N - 1, N - 2, 2**255, 2**255 - 1, 2**64, 2**64 - 1,
                 int("10" * 128, 2) % N, int("01" * 128, 2),
                 rng.randrange(1, N), rng.randrange(1, N), rng.randrange(1, N)]
    replay_pts = [(GX, "G")] + [(random_valid_x(), "rand") for _ in range(2)]
    reqs, meta = [], []
    for x, tag in replay_pts:
        for k in replay_ks:
            reqs.append(f"replay {x:x} {k:x}")
            meta.append((x, k, tag))
    out = run(reqs)
    pos = 0
    n_steps_checked = 0
    for (x, k, tag) in meta:
        head = out[pos]; pos += 1
        assert head.startswith("steps "), head
        n, fx, fy = head.split()[1:]
        n = int(n)
        g = (x, recover_even_y(x))
        osteps, ofinal = replay_schedule(k, g) if k > 1 else ([], g)
        sched = expected_schedule(k) if k > 1 else []
        check(n == len(osteps), f"replay k={k:x} {tag}: {n} steps, oracle {len(osteps)}")
        check(int(fx, 16) == ofinal[0] and int(fy, 16) == ofinal[1],
              f"replay k={k:x} {tag}: final ({fx},{fy}) != oracle {ofinal[0]:x}")
        check(ofinal[0] == x_only_mul_ints(x, k),
              f"oracle self-check k={k:x}: replay final != direct mul")
        for j in range(n):
            rnd, op, nxt, ax, ay, lam, rx, ry = out[pos].split()[1:]; pos += 1
            if j < len(osteps):
                ornd, oop, onxt, oa, olam, orr = osteps[j]
                good = (int(rnd), int(op), int(nxt)) == (ornd, oop, onxt) \
                    and (int(ax, 16), int(ay, 16)) == oa \
                    and int(lam, 16) == olam \
                    and (int(rx, 16), int(ry, 16)) == orr
                check(good, f"replay k={k:x} {tag} step {j}: repo "
                            f"({rnd},{op},{nxt},a={ax[:12]}..,l={lam[:12]}..) != oracle "
                            f"({ornd},{oop},{onxt},a={oa[0]:x}..)"[:200])
                n_steps_checked += 1
        # independent schedule statement (from bits of k only)
        check(sched == [(s[0], s[1], s[2]) for s in osteps], f"oracle schedule self-check k={k:x}")

    print(f"ANCHOR C: {'PASS' if fails == 0 else 'FAIL'} "
          f"({n_mul} mul cases, {len(err_cases)} error paths, {len(ys)} y-recoveries, "
          f"{len(meta)} replays / {n_steps_checked} steps verified, {fails} failures)")
    return 0 if fails == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
