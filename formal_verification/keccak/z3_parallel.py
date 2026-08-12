"""Parallel driver: run all 24 round UNSAT checks + 5 negative controls +
positive control concurrently, print each result as it lands."""
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from z3_verify import check_round, positive_control

BUGS = ["theta_no_rot", "rho_swap", "chi_no_not", "chi_swap", "iota_no_rc"]


def w_round(r):
    return ("round", r, str(check_round(r)))


def w_bug(bug):
    return ("bug", bug, str(check_round(1, bug=bug)))


def w_pos():
    ok, msg = positive_control(5, seed=1)
    return ("pos", ok, msg)


def main():
    with ProcessPoolExecutor(max_workers=10) as ex:
        futs = []
        futs.append(ex.submit(w_pos))
        for bug in BUGS:
            futs.append(ex.submit(w_bug, bug))
        for r in range(24):
            futs.append(ex.submit(w_round, r))

        results = {}
        for f in as_completed(futs):
            kind, key, val = f.result()
            results[(kind, key)] = val
            print(f"DONE {kind} {key} -> {val}", flush=True)

    print("\n================ SUMMARY ================", flush=True)
    # positive control stored under actual ok value; find it
    pos_ok = ("pos", True) in results
    pos_msg = results.get(("pos", True)) or results.get(("pos", False))
    print(f"positive control (non-vacuity): {'PASS' if pos_ok else 'FAIL'}  ({pos_msg})")

    neg_ok = True
    for bug in BUGS:
        v = results[("bug", bug)]
        ok = (v == "sat")
        neg_ok &= ok
        print(f"negative control {bug:14s}: {v:6s} {'OK' if ok else 'FAIL(!vacuous)'}")

    all_unsat = True
    for r in range(24):
        v = results[("round", r)]
        all_unsat &= (v == "unsat")
    bad = [r for r in range(24) if results[("round", r)] != "unsat"]
    print(f"main check: {'ALL 24 UNSAT' if all_unsat else 'NOT ALL UNSAT: ' + str([(r, results[('round', r)]) for r in bad])}")

    verdict = pos_ok and neg_ok and all_unsat
    print("\nVERDICT:", "VERIFIED (given contracts)" if verdict else "NOT VERIFIED — investigate")
    sys.exit(0 if verdict else 1)


if __name__ == "__main__":
    main()
