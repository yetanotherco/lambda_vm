#import "/book.typ": book-page, rj
#import "/src.typ": load_config, load_chip
#import "/chip.typ": (
  render_chip_assumptions,
  render_chip_variable_table,
  total_nr_variables,
  total_nr_instantiated_columns,
  render_constraint_table,
)

#let config = load_config()
#let chip = load_chip("src/bitwise.toml", config)

#let bitwise = raw(chip.name)

#show: book-page(chip.name)
#let bitwise = raw(chip.name)

The #bitwise chips deal with precomputed lookup tables for bitwise boolean operations
and convenience functionalities over small domains.

= Variables
#let nr_variables = total_nr_variables(chip)
#let nr_columns = total_nr_instantiated_columns(chip, config)
#let nr_precomputed = ("input", "output").map(c => chip.variables.at(c)).flatten().len()

The #bitwise chip is comprised of #nr_variables variables that are expressed using #nr_columns columns.
Of these, the _input_ and _output_ variables (#nr_precomputed in total) are precomputed.

#render_chip_variable_table(chip, config)

*Note*: This table contains one row for every possible value of `(X, Y, Z)`.
As such, it has length $2^8 dot 2^8 dot 2^4 = 2^(20)$.

We use the ALU operation descriptors from @decode to identify the operations in the `BYTE_ALU` interaction.
Since each of the three columns is only $2^16$ rows long, they can be combined in a single $2^20$ column (with room to spare).

= Lookup
This chip adds the following interactions to the lookup:
#render_constraint_table(chip, config)

= Notes/Optimizations
The following ideas may prove to be optimizations for the #bitwise chip:
+ Drop `MSB8` column, and instead define the `MSB8` lookup as `MSB8<X> := MSB16[256X]`.
  Note: currently, `MSB8` also implicity range checks the input `X` (the lookup fails if `X` is not a `Byte`).
  This optimization should only be executed when all chips leveraging `MSB8` do _not_ need this implicit range check.
+ Place the 16-bit (`AND`, `OR`, `XOR`, `MSB16`, etc.) and 20-bit (`HWSL`, `IS_B20`, `ZERO`) lookups in separate tables.
