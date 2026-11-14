#!/usr/bin/env bash
# Re-pin every ethrex git dependency in this repo to one rev, atomically.
#
# The guest and the host-side tooling exchange ethrex types (the rkyv stateless
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
# A guest manifest may build its ethrex crates from a fork through a [patch] section; see
# fork_patches below. --show checks the fork's recorded base is the pin, and a re-pin
# refuses while the fork sits on another rev.
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

# The pins: every lambdaclass/ethrex dependency's rev. A fork patch's revs (below) are not pins.
current_revs() {
  grep -ho 'lambdaclass/ethrex\.git", rev = "[0-9a-f]\{40\}"' "${MANIFESTS[@]}" | sort -u | sed 's/.*rev = //;s/"//g'
}

# A guest manifest may build ethrex from a fork: a `[patch."https://github.com/lambdaclass/ethrex.git"]`
# section whose entries name one git URL and one rev, with an `# ethrex-fork-base: <sha>` line naming the
# pinned rev the fork's commits sit on. The fixtures and the native reference stay on the pin, so the fork
# must keep ProgramInput as it is at its base, and its base must be the pin. Prints, for each manifest
# carrying such a patch, "<manifest> <url> <rev> <base>"; an incomplete patch prints "<manifest> INVALID <why>".
fork_patches() {
  local m section urls revs base
  for m in "${GUEST_MANIFESTS[@]}"; do
    section="$(awk '/^\[/ { p = ($0 == "[patch.\"https://github.com/lambdaclass/ethrex.git\"]") } p' "$m")"
    [[ -n "$section" ]] || continue
    urls="$(printf '%s\n' "$section" | grep -o 'git = "[^"]*"' | sort -u | sed 's/git = //;s/"//g')"
    revs="$(printf '%s\n' "$section" | grep -o 'rev = "[0-9a-f]\{40\}"' | sort -u | sed 's/rev = //;s/"//g')"
    base="$(sed -n 's/^# ethrex-fork-base: \([0-9a-f]\{40\}\)$/\1/p' "$m")"
    if [[ "$(printf '%s\n' "$urls" | grep -c .)" != "1" || "$(printf '%s\n' "$revs" | grep -c .)" != "1" ]]; then
      echo "$m INVALID the patch must name one git URL and one rev (urls: $(echo $urls), revs: $(echo $revs))"
    elif [[ "$(printf '%s\n' "$base" | grep -c .)" != "1" ]]; then
      echo "$m INVALID no single '# ethrex-fork-base: <sha>' line"
    else
      echo "$m $urls $revs $base"
    fi
  done
}

if [[ "${1:-}" == "--show" || $# -eq 0 ]]; then
  # No `mapfile`: macOS ships bash 3.2, where it does not exist.
  revs="$(current_revs)"
  if [[ "$(printf '%s\n' "$revs" | wc -l | tr -d ' ')" == "1" ]]; then
    forks="$(fork_patches)"
    if [[ -n "$forks" ]]; then
      # A fork patch is legitimate only on the pin: its base must be the pinned rev.
      while read -r manifest url rev base; do
        if [[ "$url" == "INVALID" || "$base" != "$revs" ]]; then
          echo "FORK: $manifest: ${url} ${rev} ${base} (the fork's base must be the pin $revs)" >&2
          exit 1
        fi
        echo "fork: $manifest builds ethrex from $url at $rev (base $base)" >&2
      done <<< "$forks"
    fi
    echo "$revs"
  else
    # Non-zero on purpose: this is the skew the script exists to prevent, and it is
    # what makes `--show` usable as a CI gate. A `--guest-only` re-pin is legitimately
    # this state, so a caller doing that should ignore the status.
    echo "MIXED:" >&2
    printf '  %s\n' $revs >&2
    exit 1
  fi
  exit 0
fi

COUNTS=()
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

# A fork patch sits on one ethrex rev: moving the pin under it would build the guest from the fork's old
# base while the fixtures move. Rebase the fork onto the new rev, update the patch's revs and its
# `ethrex-fork-base` line (or remove the patch) first.
while read -r manifest url rev base; do
  [[ -n "${manifest:-}" ]] || continue
  if [[ "$url" == "INVALID" || "$base" != "$NEW_REV" ]]; then
    echo "error: $manifest builds ethrex from a fork ($url $rev, base $base)." >&2
    echo "  Rebase the fork onto $NEW_REV and update its [patch] revs and ethrex-fork-base line," >&2
    echo "  or remove the patch, then re-pin. Nothing was written." >&2
    exit 1
  fi
done <<< "$(fork_patches)"

# Only rewrite revs on ethrex deps: the same file may pin other git deps.
PATTERN='(github\.com/lambdaclass/ethrex\.git", rev = ")[0-9a-f]{40}(")'

# Two passes so a partial rewrite is impossible. Pass 1 only counts: the pattern needs
# the git URL and `rev` on ONE physical line, so a manifest in `[dependencies.x]` table
# form matches nothing, and rewriting the others anyway would leave exactly the skew this
# script exists to prevent -- while reporting success.
for manifest in "${TARGETS[@]}"; do
  [[ -f "$manifest" ]] || { echo "error: missing $manifest" >&2; exit 1; }
  n="$(perl -ne "\$c++ while m{$PATTERN}g; END { print \$c + 0 }" "$manifest")"
  if [[ "$n" == "0" ]]; then
    echo "error: $manifest carries no ethrex rev this script can rewrite." >&2
    echo "  The pattern needs the git URL and \`rev\` on one line; a \`[dependencies.x]\`" >&2
    echo "  table form does not match. Nothing was written." >&2
    exit 1
  fi
  COUNTS+=("$manifest ($n)")
done

# Pass 2 rewrites, now that every target is known to match.
for manifest in "${TARGETS[@]}"; do
  perl -pi -e "s{$PATTERN}{\${1}$NEW_REV\${2}}g" "$manifest"
done

echo "re-pinned $SCOPE to $NEW_REV:"
printf '  %s\n' "${COUNTS[@]}"
