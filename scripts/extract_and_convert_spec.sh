#!/bin/bash
# Extract spec TOML files from a spec branch and convert to Markdown
#
# Usage:
#   ./scripts/extract_and_convert_spec.sh [branch] [output_dir]
#
# Default branch: origin/spec/main
# Default output directory: docs/spec

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
BRANCH="${1:-origin/spec/main}"
OUTPUT_DIR="${2:-$REPO_ROOT/docs/spec}"
TEMP_DIR=$(mktemp -d)

echo "Extracting spec files from $BRANCH..."

# Create temp directory structure
mkdir -p "$TEMP_DIR/src"

# Extract config
git show "$BRANCH:spec/src/config.toml" > "$TEMP_DIR/src/config.toml" 2>/dev/null || {
    echo "Error: Could not find spec/src/config.toml in $BRANCH"
    echo "Make sure to fetch the branch: git fetch origin <branch-name>"
    rm -rf "$TEMP_DIR"
    exit 1
}

# Extract all chip TOML files
for file in $(git ls-tree -r "$BRANCH" --name-only | grep '^spec/src/.*\.toml$' | grep -v config.toml); do
    filename=$(basename "$file")
    git show "$BRANCH:$file" > "$TEMP_DIR/src/$filename" 2>/dev/null || true
done

# Extract all Typst (.typ) files
for file in $(git ls-tree -r "$BRANCH" --name-only | grep '^spec/.*\.typ$'); do
    filename=$(basename "$file")
    git show "$BRANCH:$file" > "$TEMP_DIR/$filename" 2>/dev/null || true
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
