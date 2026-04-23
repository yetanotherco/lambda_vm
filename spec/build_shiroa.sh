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

# And build
shiroa build
