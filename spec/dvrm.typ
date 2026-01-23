#import "/book.typ": book-page
#import "/src.typ": load_config, load_chip
#import "/chip.typ": (
  render_chip_column_table,
  total_nr_variables,
  total_nr_instantiated_columns,
  render_constraint_table,
  render_chip_assumptions,
  render_chip_padding_table,
)

#let config = load_config()
// #let chip = load_chip("src/dvrm.toml", config)

#show: book-page.with(title: "DVRM chip")

*placeholder chapter: WIP*
