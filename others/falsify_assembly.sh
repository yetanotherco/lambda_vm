#!/usr/bin/env bash
# Falsification harness for the assembly leg.
#
# Applies one deliberate defect at a time to prover/src/lfm/epoch.rs, runs the
# named test, and reports PASS (test still green = the defect is INVISIBLE, a
# hole in the suite) or FAIL (the defect was caught). The verdict is read from
# the `test result:` summary line, because per-test FAILED lines do not appear
# in `cargo test -q` output — the trap the fri-emitter leg hit.
set -u
cd "$(dirname "$0")/.."
FILE=prover/src/lfm/epoch.rs
TEST=${2:-lfm::epoch_tests}
cp "$FILE" /tmp/epoch.rs.bak

restore() { cp /tmp/epoch.rs.bak "$FILE"; }
trap restore EXIT

run() {
  local label="$1"
  local out
  out=$(cargo test -p lambda-vm-prover --lib "$TEST" 2>&1 | grep "test result:")
  if echo "$out" | grep -q "FAILED"; then
    echo "CAUGHT   $label   ($out)"
  elif echo "$out" | grep -q "ok\."; then
    echo "INVISIBLE $label   ($out)"
  else
    echo "ERROR    $label   (build failure or no result: $out)"
  fi
  restore
}

case "${1:-all}" in
  fri_order)
    python3 - <<'PY'
p='prover/src/lfm/epoch.rs'; s=open(p).read()
s=s.replace("""        zetas.push(t.sample_ext(b));
        t.append_halves(&root.halves());""","""        t.append_halves(&root.halves());
        zetas.push(t.sample_ext(b));""")
open(p,'w').write(s)
PY
    run "FRI: absorb the layer root BEFORE sampling its zeta"
    ;;
  fri_drop_root)
    python3 - <<'PY'
p='prover/src/lfm/epoch.rs'; s=open(p).read()
s=s.replace("""        zetas.push(t.sample_ext(b));
        t.append_halves(&root.halves());""","""        zetas.push(t.sample_ext(b));
        let _ = root;""")
open(p,'w').write(s)
PY
    run "FRI: never absorb the committed layer roots"
    ;;
  fri_drop_final)
    python3 - <<'PY'
p='prover/src/lfm/epoch.rs'; s=open(p).read()
s=s.replace("""    if shape.fri.total_folds() > 0 {
        zetas.push(t.sample_ext(b));
    }""","""    if false {
        zetas.push(t.sample_ext(b));
    }""")
open(p,'w').write(s)
PY
    run "FRI: skip the final-fold zeta draw"
    ;;
  fri_drop_coeffs)
    python3 - <<'PY'
p='prover/src/lfm/epoch.rs'; s=open(p).read()
s=s.replace("""    for c in absorbs.fri_coeffs {
        append_ext_cell(b, t, *c);
    }""","""    for c in absorbs.fri_coeffs {
        let _ = c;
    }""")
open(p,'w').write(s)
PY
    run "FRI: never absorb the terminal polynomial coefficients"
    ;;
  ood_row_major)
    python3 - <<'PY'
p='prover/src/lfm/epoch.rs'; s=open(p).read()
s=s.replace("""        for col in 0..width {
            for row in 0..height {
                append_ext_cell(b, t, block[row * width + col]);
            }
        }""","""        for row in 0..height {
            for col in 0..width {
                append_ext_cell(b, t, block[row * width + col]);
            }
        }""")
open(p,'w').write(s)
PY
    run "Round 3: absorb the OOD blocks ROW-major"
    ;;
  ood_order)
    python3 - <<'PY'
p='prover/src/lfm/epoch.rs'; s=open(p).read()
s=s.replace("""        (shape.ood_current_dims, absorbs.ood_current),
        (shape.ood_next_dims, absorbs.ood_next),""","""        (shape.ood_next_dims, absorbs.ood_next),
        (shape.ood_current_dims, absorbs.ood_current),""")
open(p,'w').write(s)
PY
    run "Round 3: absorb the next-row OOD block before the current-row one"
    ;;
  nonce_absorb)
    python3 - <<'PY'
p='prover/src/lfm/epoch.rs'; s=open(p).read()
s=s.replace("""        emit_grinding_check(b, seed, halves, shape.grinding_factor);
        t.append_halves(&halves);""","""        emit_grinding_check(b, seed, halves, shape.grinding_factor);""")
open(p,'w').write(s)
PY
    run "Grinding: never absorb the nonce"
    ;;
  grinding_check)
    python3 - <<'PY'
p='prover/src/lfm/epoch.rs'; s=open(p).read()
s=s.replace("""        emit_grinding_check(b, seed, halves, shape.grinding_factor);
        t.append_halves(&halves);""","""        let _ = seed;
        t.append_halves(&halves);""")
open(p,'w').write(s)
PY
    run "Grinding: emit no proof-of-work check at all"
    ;;
  z_guard)
    python3 - <<'PY'
p='prover/src/lfm/epoch.rs'; s=open(p).read()
s=s.replace("""    let one = b.ext_const(&FEE::one());
    assert_ne_ext(b, z_pow_trace, one);""","""    let one = b.ext_const(&FEE::one());
    let _ = one;""")
open(p,'w').write(s)
PY
    run "z_ood: drop the trace-domain non-membership guard"
    ;;
  fork_separator)
    python3 - <<'PY'
p='prover/src/lfm/epoch.rs'; s=open(p).read()
s=s.replace("""    if num_tables > 1 {
        fork.append_const_bytes(&(index as u64).to_le_bytes());
    }""","""    if false {
        fork.append_const_bytes(&(index as u64).to_le_bytes());
    }""")
open(p,'w').write(s)
PY
    run "Fork: omit the per-table domain separator"
    ;;
  contribution)
    python3 - <<'PY'
p='prover/src/lfm/epoch.rs'; s=open(p).read()
s=s.replace("""    if let Some(l) = absorbs.contribution {
        append_ext_cell(b, t, l);
    }""","""    if let Some(l) = absorbs.contribution {
        let _ = l;
    }""")
open(p,'w').write(s)
PY
    run "Phase C: never absorb the bus contribution L"
    ;;
  aux_root)
    python3 - <<'PY'
p='prover/src/lfm/epoch.rs'; s=open(p).read()
s=s.replace("""    if let Some(root) = absorbs.aux_root {
        t.append_halves(&root.halves());
    }""","""    if let Some(root) = absorbs.aux_root {
        let _ = root;
    }""")
open(p,'w').write(s)
PY
    run "Phase C: never absorb the aux trace root"
    ;;
  *)
    echo "usage: $0 <defect> [test-filter]"
    echo "defects: fri_order fri_drop_root fri_drop_final fri_drop_coeffs ood_row_major"
    echo "         ood_order nonce_absorb grinding_check z_guard fork_separator"
    echo "         contribution aux_root"
    ;;
esac
