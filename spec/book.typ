#import "@preview/shiroa:0.3.1": *

#show: book

#let meta = (
  title: "Lambda VM specification",
  authors: ("3MI Labs", "Aligned"),
  summary: (
    ("memory.typ", [Memory argument], <memory>),
    ("variables.typ", [Variables], <vars>),
    ("is_bit.typ", [IS_BIT template], <isbit>),
    ("add.typ", [ADD/SUB template], <add>),
    ("decode.typ", [DECODE table], <decode>),
    ("cpu.typ", [CPU chip], <cpu>),
    ("shift.typ", [SHIFT chip], <shift>),
    ("branch.typ", [BRANCH chip], <branch>),
    ("memw.typ", [MEMW chip], <memw>),
    ("lt.typ", [LT chip], <lt>),
    ("mul.typ", [MUL chip], <mul>),
    ("dvrm.typ", [DVRM chip], <dvrm>),
    ("load.typ", [LOAD chip], <load>),
    ("ecall.typ", [ECALL chips], <ecall>),
    ("bitwise.typ", [BITWISE chips], <bitwise>),
  )
)
#book-meta(
  title: meta.title,
  authors: meta.authors,
  summary: meta.summary.map(((ch, title, _ref)) => chapter(ch, title)).join()
)

#let is-shiroa = "x-target" in sys.inputs

#import "/templates/page.typ": project
#let book-page(file, ..args) = if is-shiroa {
  (body) => project.with(..args, title: meta.summary.find(x => x.at(0) == file).at(1))(body)
} else {
  (body) => body
}

#let todo(background: white, foreground: black, name: none, body) = block(fill: background, outset: 0.5em, radius: 20%, stroke: black)[
  #set text(fill: foreground)
  *TODO #if name != none { [(#name)] }*: #body
]
#let rj = todo.with(background: teal, name: "Robin")
#let et = todo.with(background: rgb("d4aa3a"), name: "Erik")

#let style = state("style", (
  foreground: white,
))

#let aside(title, body) = context figure(
  block(inset: (left: 1em, right: 1em, bottom: 1em), stroke: style.final().foreground, breakable: false)[
    #block(inset: (left: 1em, right: 1em, top: .75em, bottom: .75em),
           width: 100% + 2em,
           fill: rgb("55aaff"),
           stroke: style.final().foreground,
           align(center, strong(text(fill: black, title))))
    #align(left, body)
])

#show figure: repr

#let _xref-included = state("_xref-included", (:))

#let strip-all(content) = {
  if repr(content.func()) == "sequence" {
    for c in content.children {
      strip-all(c)
    }
  } else if repr(content.func()) == "styled" {
    strip-all(content.child)
  } else {
    content
  }
}

#let xref(lbl, ..ref-args) = {
  if is-shiroa {
    let found =  meta.summary.find(x => str(lbl).starts-with(str(x.at(2))))
    let (ch, title, ref) = found
    if ref == lbl {
      cross-link("/" + ch, [Chapter #(meta.summary.position(x => x == found) + 1)])
    } else {
      // Because shiroa does weird url escaping
      let shiroa-label = label(str(lbl).replace(":", "%3A"))
      context if ch not in _xref-included.get() {
        hide(box(width: 0%, height: 0%, strip-all(include "/" + ch)))
        _xref-included.update(it => it + ((ch): true))
      }
      let link-content = context {
        let fig = query(lbl).first()
        let counter = if fig.has("counter") {
          fig.counter
        } else {
          counter(fig.func())
        }

        [#ref-args.named().at("supplement", default: [])#numbering(fig.numbering, ..counter.at(lbl))]
      }
      cross-link("/" + ch, reference: shiroa-label, link-content)
    }
  } else {
    ref(lbl)
  }
}
