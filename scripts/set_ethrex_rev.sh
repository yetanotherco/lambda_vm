#!/usr/bin/env bash
# Re-pin every ethrex git dependency in this repo to one rev, atomically.
#
# The guest and the host-side tooling exchange ethrex types (the SSZ stateless
# input the converter writes and the guest decodes), so a rev that differs
# between them is not a version skew that fails to build -- it is a fixture the
# guest silently decodes as the all-zero default. Keeping every pin on one rev
# is what makes that impossible, which is why this rewrites all of them together
# rather than leaving them to be bumped by hand.
#
# Usage:
#   scripts/set_ethrex_rev.sh <40-char-sha>              re-pin everything
#   scripts/set_ethrex_rev.sh --guest-only <40-char-sha> re-pin only the guest graph
#   scripts/set_ethrex_rev.sh --show                     print the current pin(s)
#
# `--guest-only` moves the crates the guest ELF is built from and leaves the
# host-side fixture tooling where it is. That split is what lets a benchmark vary
# the guest across two revs while holding the fixture constant: the fixture is
# just bytes, its wire format does not change within a PR, and regenerating it per
# rev would instead drag the whole ethrex HOST api (which does drift) into the
# measurement.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# Every manifest carrying an ethrex git dep. Kept explicit rather than globbed so
# a new one has to be added deliberately -- a manifest silently left behind is
# exactly the skew described above.
# The guest ELF's own graph. `crypto/ethrex-crypto` is a path dep of BOTH the
# guest and the converter, so moving it moves what the converter links too --
# which is why a `--guest-only` run must come after the fixture already exists.
GUEST_MANIFESTS=(
  executor/programs/rust/ethrex/Cargo.toml
  crypto/ethrex-crypto/Cargo.toml
)

# Host-side tooling: builds fixtures, never lands in the ELF.
TOOLING_MANIFESTS=(
  tooling/ethrex-tests/Cargo.toml
  tooling/ethrex-block-converter/Cargo.toml
  tooling/ethrex-fixtures/Cargo.toml
)

MANIFESTS=("${GUEST_MANIFESTS[@]}" "${TOOLING_MANIFESTS[@]}")

current_revs() {
  grep -ho 'rev = "[0-9a-f]\{40\}"' "${MANIFESTS[@]}" | sort -u | sed 's/rev = //;s/"//g'
}

if [[ "${1:-}" == "--show" || $# -eq 0 ]]; then
  # No `mapfile`: macOS ships bash 3.2, where it does not exist.
  revs="$(current_revs)"
  if [[ "$(printf '%s\n' "$revs" | wc -l | tr -d ' ')" == "1" ]]; then
    echo "$revs"
  else
    # Expected mid-benchmark (a `--guest-only` re-pin is exactly this state), so
    # report it rather than failing.
    echo "MIXED:" >&2
    printf '  %s\n' $revs >&2
    exit 1
  fi
  exit 0
fi

TARGETS=("${MANIFESTS[@]}")
SCOPE="all manifests"
if [[ "${1:-}" == "--guest-only" ]]; then
  shift
  TARGETS=("${GUEST_MANIFESTS[@]}")
  SCOPE="the guest graph only"
fi

NEW_REV="${1:-}"
if [[ ! "$NEW_REV" =~ ^[0-9a-f]{40}$ ]]; then
  echo "error: expected a 40-character commit sha, got '$NEW_REV'" >&2
  exit 2
fi

for manifest in "${TARGETS[@]}"; do
  [[ -f "$manifest" ]] || { echo "error: missing $manifest" >&2; exit 1; }
  # Only rewrite revs on ethrex deps: the same file may pin other git deps.
  perl -pi -e 's{(github\.com/lambdaclass/ethrex\.git", rev = ")[0-9a-f]{40}(")}{${1}'"$NEW_REV"'${2}}g' "$manifest"
done

echo "re-pinned $SCOPE to $NEW_REV:"
printf '  %s\n' "${TARGETS[@]}"
