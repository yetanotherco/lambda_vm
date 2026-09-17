#!/usr/bin/env bash
# Exercises scripts/assert_workload_cycles.sh, which is the only thing standing between a
# benchmark and a workload the guest rejected. Every branch of it is a hard stop, so a
# typo in one is a gate that passes everything -- and the gate is shell, which no other
# test in this repo would notice.
#
# Fake CLIs instead of the real one: the property under test is the decision, not the
# executor. Real runs of both floors are in the commit that introduced them (a pre-SSZ
# fixture at 496 cycles against 37,137,748 for the real block).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
GUARD="$ROOT/scripts/assert_workload_cycles.sh"
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT
failures=0

fake_cli() {  # $1=name $2=body
  printf '#!/usr/bin/env bash\n%s\n' "$2" > "$WORK/$1"
  chmod +x "$WORK/$1"
  echo "$WORK/$1"
}

check() {  # $1=label $2=expected status $3.. = command
  local label="$1" want="$2"; shift 2
  local out status
  out="$("$@" 2>&1)" && status=0 || status=$?
  if [ "$status" != "$want" ]; then
    echo "FAIL  $label: exit $status, expected $want"
    printf '%s\n' "$out" | sed 's/^/      /'
    failures=$((failures + 1))
  else
    echo "ok    $label (exit $status)"
  fi
}

counts_37m="$(fake_cli cli_37m 'echo "Cycles: 37137748"')"
counts_496="$(fake_cli cli_496 'echo "Cycles: 496"')"
counts_1m="$(fake_cli cli_1m 'echo "Cycles: 1000000"')"
counts_none="$(fake_cli cli_none 'echo "Execution failed: UnknownSyscall(18446744073709551615)" >&2; exit 1')"

check "real block passes the real floor"          0 "$GUARD" "$counts_37m"  elf input real
check "rejected input fails the real floor"       1 "$GUARD" "$counts_496"  elf input real
check "rejected input fails the synthetic floor"  1 "$GUARD" "$counts_496"  elf input synthetic
check "1M passes synthetic but fails real"        0 "$GUARD" "$counts_1m"   elf input synthetic
check "  (same count, real floor)"                1 "$GUARD" "$counts_1m"   elf input real
check "a run with no cycle count is not a pass"   1 "$GUARD" "$counts_none" elf input real
check "an unknown workload is a usage error"      2 "$GUARD" "$counts_37m"  elf input realish
check "too few arguments are a usage error"       2 "$GUARD" "$counts_37m"  elf input
check "--floor prints the real floor"             0 "$GUARD" --floor real
check "--floor rejects an unknown workload"       2 "$GUARD" --floor nonsense

# The floors are the reason the pass/fail pairs above land where they do, so a change to
# either without a change here would leave the pairs meaningless.
real_floor="$("$GUARD" --floor real)"
synth_floor="$("$GUARD" --floor synthetic)"
if [ "$real_floor" != "20000000" ] || [ "$synth_floor" != "100000" ]; then
  echo "FAIL  floors moved: real=$real_floor synthetic=$synth_floor (test pinned 20000000/100000)"
  failures=$((failures + 1))
else
  echo "ok    floors are 20000000 / 100000"
fi

# And the diagnostic has to name the fault: "no cycle count" and "below the floor" are
# different repairs, and the first one used to be reported as the second.
# Captured first: the guard exits non-zero by design, and `set -o pipefail` would make
# the pipeline's status the guard's rather than grep's.
diag="$("$GUARD" "$counts_none" elf input real 2>&1 || true)"
if ! printf '%s\n' "$diag" | grep -q "printed no cycle count"; then
  echo "FAIL  an execution error is not reported as one"
  failures=$((failures + 1))
else
  echo "ok    an execution error reports the output, not a rejection"
fi

if [ "$failures" != "0" ]; then
  echo "$failures check(s) failed"
  exit 1
fi
echo "all checks passed"
