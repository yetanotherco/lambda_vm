# Lambda VM Specification

Formal specification of the Lambda VM. Covers the per-chip AIR constraints (CPU, decode, bitwise, branch, LT, shift, MUL, DVRM, MEMW, LOAD, page, register, halt, commit, keccak), the memory argument, and the LogUp lookup framework that links the tables.

The specification is written in [Typst](https://typst.app/) and rendered as either a PDF or a browsable bundle of web pages [Typst's HTML export](https://typst.app/docs/reference/bundle/).

## Rendering it locally

1. [Install Typst](https://github.com/typst/typst?tab=readme-ov-file#installation).
2. From this directory, run:

   ```sh
   typst compile spec.typ
   ```
   to compile the spec as a PDF (`spec.pdf`), or
   ```sh
   typst compile --features bundle,html --format bundle bundle.typ
   ```
   to compile the web format to `bundle/`.
