# LambdaVM specification
This repository contains specification for [`LambdaVM`](https://github.com/yetanotherco/lambda_vm).
The specification is written in [`Typst`](https://typst.app/) and can be rendered by [`shiroa`](https://myriad-dreamin.github.io/shiroa/) as either a file (pdf) or a wiki (html).

## Installation & Development setup
1. [Install `Typst`](https://github.com/typst/typst?tab=readme-ov-file#installation).
2. [Install `shiroa`](https://myriad-dreamin.github.io/shiroa/guide/installation.html).
3. Clone this repository.
4. Open the repository in a terminal and execute `shiroa serve`.

At this point, the wiki version is hosted locally and is actively updated as you modify the specification files.

## Converting to Markdown

To convert the spec TOML files to Markdown (for documentation or LLM consumption):

```bash
# From the repository root:
python3 scripts/spec_to_md.py spec/src/config.toml spec/src/*.toml

# Or output to a specific directory:
python3 scripts/spec_to_md.py --output-dir docs/spec spec/src/config.toml spec/src/*.toml
```

This generates one Markdown file per chip (cpu.md, add.md, lt.md, etc.) with tables for columns, constraints, and assumptions.
