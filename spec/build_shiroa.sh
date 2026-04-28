#!/usr/bin/env bash

set -euo pipefail

# cd into the script directory
cd "$(dirname "${BASH_SOURCE[0]}")"

# Clean up potential old file
rm -f interaction_count.json

# Always clean up after ourselves
trap 'rm -f interaction_count.json' EXIT

# Query the ebook version for the proper counts
typst query ebook.typ '<interaction_count>' --field value > interaction_count.json

# Check if there's enough memory available for a parallel shiroa build
# 20GiB as comfortable baseline
available_kb=$(awk '/MemAvailable/ { print $2 }' /proc/meminfo)
required_kb=$((20 * 1024 * 1024))
if [ "$available_kb" -lt "$required_kb" ]; then
  echo "Falling back to single-thread"
  export RAYON_NUM_THREADS=1
fi

# And build
shiroa build
