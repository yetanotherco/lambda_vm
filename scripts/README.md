# Scripts

## typst_to_md.py

Converts the Typst specification to Markdown format.

### What it does

1. Parses `.typ` files for prose content (headings, paragraphs, notes)
2. Parses `.toml` files for structured data (variables, constraints, assumptions)
3. Detects `#render_constraint_table()` calls to insert tables at correct positions
4. Reads constraint group prefixes from TOML (e.g., `prefix = "R"` → `CR`)
5. Maintains continuous constraint numbering across groups (CPU-C1 → CPU-CR2 → ...)

### Usage

```bash
cd scripts
source .venv/bin/activate
python typst_to_md.py                          # Output to ../docs/spec/
python typst_to_md.py -o ../others/spec_md     # Output to specific directory
```

### Requirements

Python 3.8+ with `tomli` (or Python 3.11+ which has `tomllib` built-in):

```bash
cd scripts
python -m venv .venv
source .venv/bin/activate
pip install tomli
```

### Output

Generates 16 markdown files:
- Individual chapter files (`cpu.md`, `memw.md`, etc.)
- Combined file (`spec_full.md`)

### Notes

- Math expressions are preserved in Typst notation (not LaTeX), but semantically equivalent
- The script reads from `../spec/` (typst source) and `../spec/src/` (TOML data)
