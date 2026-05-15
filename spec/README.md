# Lambda VM Specification

Formal specification of the Lambda VM. Covers the per-chip AIR constraints (CPU, decode, bitwise, branch, LT, shift, MUL, DVRM, MEMW, LOAD, page, register, halt, commit, keccak), the memory argument, and the LogUp lookup framework that links the tables.

The specification is written in [Typst](https://typst.app/) and rendered as either a PDF or a browsable HTML wiki using [shiroa](https://myriad-dreamin.github.io/shiroa/).

## Rendering it locally

1. [Install Typst](https://github.com/typst/typst?tab=readme-ov-file#installation).
2. [Install shiroa](https://myriad-dreamin.github.io/shiroa/guide/installation.html).
3. From this directory, run:

   ```sh
   shiroa serve
   ```

   shiroa will host the HTML wiki locally and live-reload as you edit the `.typ` source files.

To produce a PDF instead, see the shiroa documentation for the `build` command.
