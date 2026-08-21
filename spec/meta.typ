#import "equate-lite.typ": equate

#let meta = (
  title: "Lambda VM specification",
  authors: ("3MI Labs", "Aligned"),
  version: "0.2",
  summary: (
    ("PROOF SYSTEM", (
        ("logup", [`LogUp` argument], <logup>),
        ("memory", [Memory argument], <memory>),
        ("streaming", [Streaming prover], <streaming>),
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
      ("reg", [`REG`/`REGW` template], <reg>),
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
      ("ecsm", [`ECSM` accelerator], <ecsm>),
      ("fext", [Extension field accelerator], <fext>),
    )),
    ("MATHEMATICS", (
      ("limbs_and_carries", [On limb decomposition and carries], <limbs>),
    ))
  )
)

#let todo(background: white, foreground: black, name: none, body) = block(fill: background, outset: 0.4em, radius: 20%, stroke: black)[
  #set text(fill: foreground)
  *TODO #if name != none { [(#name)] }*: #body
]

#let rj = todo.with(background: teal, name: "Robin")
#let et = todo.with(background: rgb("d4aa3a"), name: "Erik")
#let cdsg = todo.with(background: olive, name: "Cyprien")


#let highlights = (
  "aside": ("Aside", rgb("55aaff")),
  "attention": ("Attention", rgb("ff2600")),
)

#let highlight(title, body, ref: none, kind: "aside") = [
  #figure(
    caption: title,
    supplement: highlights.at(kind).at(0),
    kind: kind,
    body
  )#ref
]

#let aside = highlight.with(kind: "aside")
#let attention = highlight.with(kind: "attention")

#let stripe_tables(body) = context if target() == "html" {
  show table: it => html.div(class: "striped", it)
  body
} else if target() == "paged" {
  set table(fill: (_, y) => if calc.odd(y) { color.rgb(0, 0, 100, 20) } else { color.rgb(255, 255, 255, 20) })
  body
} else {
  assert(false, message: "Unsupported target: " + target())
}

#let common-formatting(body) = {
  set footnote(numbering: "[1]")
  show raw.where(block: true): it => block(it, inset: 1em, width: 100%, radius: 5pt)
  show ref: equate.with(sub-numbering: true, breakable: true, number-mode: "label")
  show selector.or(..highlights.keys().map(k => figure.where(kind: k))): it => {
    set figure.caption(position: top)
    show figure.caption: cap => block(
      inset: (left: 1em, right: 1em, top: .75em, bottom: .75em),
      outset: (left: 1em),
      width: 100% + 1em,
      fill: highlights.at(it.kind).at(1),
      stroke: luma(50%),
      align(center, strong(text(fill: black, cap)))
    )
    block(inset: (left: 1em, right: 1em, bottom: 1em), stroke: luma(50%), breakable: false, align(left, it))
  }
  body
}
