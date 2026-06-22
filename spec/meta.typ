#let meta = (
  title: "Lambda VM specification",
  authors: ("3MI Labs", "Aligned"),
  version: "0.2",
  summary: (
    ("PROOF SYSTEM", (
        ("logup", [`LogUp` argument], <logup>),
        ("memory", [Memory argument], <memory>),
    )),
    ("OVERVIEW", (
        ("variables", [Variables], <vars>),
        ("signatures", [Signatures], <signatures>),
    )),
    ("TEMPLATES", (
      ("is_bit", [`IS_BIT` template], <isbit>),
      ("is_byte", [`IS_BYTE` template], <isbyte>),
      ("sign", [`SIGN` template], <sign>),
      ("add", [`ADD`/`SUB` template], <add>),
      ("neg", [`NEG` template], <neg>),
    )),
    ("CPU", (
      ("decode", [`DECODE` table], <decode>),
      ("cpu", [`CPU` chip], <cpu>),
      ("cpu32", [`CPU32` chip], <cpu32>),
    )),
    ("ALU", (
      ("shift", [`SHIFT` chip], <shift>),
      ("branch", [`BRANCH` chip], <branch>),
      ("lt", [`LT` chip], <lt>),
      ("eq", [`EQ` chip], <eq>),
      ("mul", [`MUL` chip], <mul>),
      ("dvrm", [`DVRM` chip], <dvrm>),
      ("bitwise", [`BITWISE` chips], <bitwise>),
      ("bytewise", [`BYTEWISE` chip], <bytewise>)
    )),
    ("MEMORY", (
      ("memw", [`MEMW` chip], <memw>),
      ("load", [`LOAD` chip], <load>),
      ("store", [`STORE` chip], <store>),
    )),
    ("ECALLS", (
      ("about_ecalls", [About `ECALL`], <ecall>),
      ("halt", [`HALT` chip], <halt>),
      ("commit", [`COMMIT` chip], <commit>),
      ("sha256", [`SHA256` accelerator], <sha256>),
      ("keccak", [`KECCAK` accelerator], <keccak>),
    ))
  )
)

#let common-formatting(body) = {
  set footnote(numbering: "[1]")
  show raw.where(block: true): it => block(it, inset: 1em, width: 100%, radius: 5pt)
  body
}

#let todo(background: white, foreground: black, name: none, body) = block(fill: background, outset: 0.4em, radius: 20%, stroke: black)[
  #set text(fill: foreground)
  *TODO #if name != none { [(#name)] }*: #body
]

#let rj = todo.with(background: teal, name: "Robin")
#let et = todo.with(background: rgb("d4aa3a"), name: "Erik")
#let cdsg = todo.with(background: olive, name: "Cyprien")

#let aside(title, body) = context figure(
  block(inset: (left: 1em, right: 1em, bottom: 1em), stroke: luma(50%), breakable: false)[
    #block(inset: (left: 1em, right: 1em, top: .75em, bottom: .75em),
           width: 100% + 2em,
           fill: rgb("55aaff"),
           stroke: luma(50%),
           align(center, strong(text(fill: black, title))))
    #align(left, body)
])
