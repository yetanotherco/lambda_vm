#import "@preview/shiroa:0.3.1": *

#show: book

#book-meta(
  title: "Lambda VM specification",
  summary: [
    #chapter("memory.typ")[Memory argument]
    #chapter("variables.typ")[Variables]
    #chapter("is_bit.typ")[IS_BIT template]
    #chapter("add.typ")[ADD template]
    #chapter("decode.typ")[DECODE chip]
    #chapter("cpu.typ")[CPU chip]
    #chapter("shift.typ")[SHIFT chip]
    #chapter("branch.typ")[BRANCH]
    #chapter("memw.typ")[MEMW]
    #chapter("lt.typ")[LT]
    #chapter("mul.typ")[MUL chip]
    #chapter("dvrm.typ")[DVRM chip]
    #chapter("load.typ")[LOAD chip]
    #chapter("ecall.typ")[ECALL chips]
    #chapter("bitwise.typ")[BITWISE]
  ]
)

// re-export page template
#import "/templates/page.typ": project
#let book-page = project

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

#let is-shiroa = is-web-target()

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

#let xref(file, lbl, ..ref-args) = {
  if is-shiroa {
    // Because shiroa does weird url escaping
    let shiroa-label = label(str(lbl).replace(":", "%3A"))
    context if file not in _xref-included.get() {
        // Let's blow up the compile times :)
        hide(box(width: 0%, height: 0%, strip-all(include file)))
        _xref-included.update(it => it + ((file): true))
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
    cross-link(file, reference: shiroa-label, link-content)
  } else {
    ref(lbl, ..ref-args)
  }
}
