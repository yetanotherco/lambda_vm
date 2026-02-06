#!/bin/bash
# Extract spec TOML files from spec/main branch and convert to Markdown
#
# Usage:
#   ./scripts/extract_and_convert_spec.sh [output_dir]
#
# Default output directory: docs/spec

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
OUTPUT_DIR="${1:-$REPO_ROOT/docs/spec}"
TEMP_DIR=$(mktemp -d)

echo "Extracting spec files from origin/spec/main..."

# Create temp directory structure
mkdir -p "$TEMP_DIR/src"

# Extract config
git show origin/spec/main:spec/src/config.toml > "$TEMP_DIR/src/config.toml" 2>/dev/null || {
    echo "Error: Could not find spec/src/config.toml in origin/spec/main"
    echo "Make sure to fetch the branch: git fetch origin spec/main"
    rm -rf "$TEMP_DIR"
    exit 1
}

# Extract all chip TOML files
for file in $(git ls-tree -r origin/spec/main --name-only | grep '^spec/src/.*\.toml$' | grep -v config.toml | grep -v page.toml); do
    filename=$(basename "$file")
    git show "origin/spec/main:$file" > "$TEMP_DIR/src/$filename" 2>/dev/null || true
done

# Extract all Typst (.typ) files
for file in $(git ls-tree -r origin/spec/main --name-only | grep '^spec/.*\.typ$'); do
    filename=$(basename "$file")
    git show "origin/spec/main:$file" > "$TEMP_DIR/$filename" 2>/dev/null || true
done

# List extracted files
echo "Extracted files:"
ls -la "$TEMP_DIR/src/"
echo ""
echo "Extracted .typ files:"
ls -la "$TEMP_DIR/"*.typ 2>/dev/null || echo "(none)"

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Run the Python converter
echo ""
echo "Converting to Markdown..."
python3 "$SCRIPT_DIR/typst_to_md.py" \
    --spec-dir "$TEMP_DIR" \
    --output-dir "$OUTPUT_DIR"

# Cleanup
rm -rf "$TEMP_DIR"

echo ""
echo "Done! Markdown files written to: $OUTPUT_DIR"
ls -la "$OUTPUT_DIR"
