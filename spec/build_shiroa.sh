#!/usr/bin/env bash

set -euo pipefail

typst query ebook.typ '<interaction_count>' --field value > interaction_count.json
shiroa build
rm -f interaction_count.json
