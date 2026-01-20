#import "/book.typ": book-page, rj
#import "/src.typ": load_config, load_chip
#import "/chip.typ": (
  render_chip_assumptions,
  render_chip_column_table,
  total_nr_variables,
  total_nr_instantiated_columns,
  render_constraint_table,
  render_chip_padding_table,
)

#let config = load_config()
#let chip = load_chip("src/decode.toml", config)
#show: book-page.with(title: "DECODE chip")

#let decode = raw(chip.name)

= #decode table
All `RV64ACIM` instruction are to be decoded to a format that can be interpreted by the VM.
This section outlines the decoding table being used in the VM.
For reasons of efficiency, data in this table is significantly compressed.
Since reasoning about this compressed form is needlessly complex, the `decode (uncompressed)` section presents the same table in uncompressed form, and explains how to decode `RV64IM` assembly instructions to it.
Instructions on how to compress the uncompressed table to form the compressed decode table, can be derived from the `packed_decode` variable provided below.

== Columns
#let nr_variables = total_nr_variables(chip)
#let nr_columns = total_nr_instantiated_columns(chip, config)

The #decode table is comprised of #nr_variables variables that are expressed using #nr_columns columns:
#render_chip_column_table(chip, config)

== Padding
The #decode table must be padded to a length that is a power of two.
Empty rows with the following content can be added to achieve this:
#render_chip_padding_table(chip, config)
