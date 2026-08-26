#import "/src.typ": load_config

#let config = load_config()

While this VM operates on 64-bit words, the proving system's base field has fewer than $2^64$ elements available and thus cannot represent all words natively.
To this end, we introduce the concept of "variables" as an abstraction layer on top of the VM's field elements. The following table lists all variable types used in this VM.

#table(
  columns: (auto, 1fr, auto),
  inset: 7pt,
  align: (top+left, top+left, top+center, ),
  table.header([*Name*], [*Description*], [*\#Columns*]),
  ..for type in config.variables.types {
    ([#raw(type.label)], [#eval(type.desc, mode: "markup")], [#type.subtypes.len()])
  },
)
