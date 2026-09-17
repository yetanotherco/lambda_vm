#import "/src.typ": load_config, load_chip
#import "/chip.typ": render_chip_variable_table, render_constraint_table
#let config = load_config()
#let chip = load_chip("/src/field_vm_decode.toml", config)
#let decode = raw(chip.name)

In this chapter, we provide a brief overview of the #decode chip,
that corresponds to the instruction decoding for @field-VM.
As the ISA from @field-VM:sec:isa was designed to have a simple mapping
onto AIR tables, the decoding table is itself also simple.

We present the table in its uncompressed form, but in practice, any
implementation would materialize the _virtual_ and _multiplicity_ columns only,
similar to the approach in @decode.
Due to its relative simplicity, we do not present both compressed and uncompressed
variants of the table separately.

= Variables

#render_chip_variable_table(chip, config)

= Constraints

#render_constraint_table(chip, config)
