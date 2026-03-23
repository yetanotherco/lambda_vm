#import "@preview/shiroa:0.3.1": *
#import "/templates/page.typ": project

#show: book

#let meta = (
  title: "Lambda VM specification",
  authors: ("3MI Labs", "Aligned"),
  version: "0.2",
  summary: (
    ("logup.typ", [LogUp argument], <logup>),
    ("memory.typ", [Memory argument], <memory>),
    ("variables.typ", [Variables], <vars>),
    ("signatures.typ", [Signatures], <signatures>),
    ("is_bit.typ", [IS_BIT template], <isbit>),
    ("sign.typ", [SIGN template], <sign>),
    ("add.typ", [ADD/SUB template], <add>),
    ("neg.typ", [NEG template], <neg>),
    ("decode.typ", [DECODE table], <decode>),
    ("cpu.typ", [CPU chip], <cpu>),
    ("shift.typ", [SHIFT chip], <shift>),
    ("branch.typ", [BRANCH chip], <branch>),
    ("memw.typ", [MEMW chip], <memw>),
    ("regw.typ", [REGW chip], <regw>),
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
  summary: prefix-chapter("front.typ", meta.title) + meta.summary.map(((ch, title, _ref)) => chapter(ch, title)).join()
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


#let is-shiroa = "x-target" in sys.inputs

// Strip styling to keep only "pure" content.
// This is useful to avoid errors on the `set document(...)` in `project`
// when invisibly including other chapters to resolve xrefs.
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

#let _toplevel = state("_toplevel", none)
#let _xref-included = state("_xref-included", (:))

// Invisibly include another chapter, so that its labels can be resolved
#let xref-include(f) = {
  context {
    place(hide(box(width: auto, height: 0%, strip-all(include "/" + f))))
  }
}

// Generate a cross-link for references to other chapters.
// Leaves the ref untouched if it can't be resolved or points to the current chapter.
#let xref(rf) = {
  assert(is-shiroa, message: "xref should only be used when compiling for shiroa")
  let lbl = rf.target
  let found = meta.summary.find(((_, _, tag)) => str(lbl).starts-with(str(tag)))
  context if found != none and found.at(0) != _toplevel.final() {
    let (ch, title, ref) = found
    if ref == lbl {
      cross-link("/" + ch, [Chapter #(meta.summary.position(x => x == found) + 1)])
    } else {
      // Because shiroa does weird url escaping
      let shiroa-label = label(str(lbl).replace(":", "%3A"))
      context _xref-included.update(x => x + ((ch): true))
      // The ideal would be to use `rf` directly as content argument to `cross-link`,
      // as that would inherit any/all formatting of the ref we want or need.
      // Unfortunately the ref link seems to take precedence over the cross-link hyperlink
      // when clicking.
      // There may still be some way around it by messing with some html output
      let link-content = context {
        let fig = query(lbl).first()
        let counter = if fig.has("counter") {
          fig.counter
        } else {
          counter(fig.func())
        }

        let supplement = if rf.supplement == auto {
          fig.fields().at("supplement", default: none)
        } else {
          rf.supplement
        }
        [#supplement#numbering(fig.numbering, ..counter.at(lbl))]
      }
      cross-link("/" + ch, reference: shiroa-label, link-content)
    }
  } else {
    rf
  }
}

#let book-page(file, ..args) = {
  let file = if file.ends-with(".typ") {
    file
  } else {
    lower(file) + ".typ"
  }
  assert(meta.summary.find(((f, _, _)) => f == file) != none, message: "Couldn't resolve typst source file " + file)
  if is-shiroa {
    (body) => {
      show: common-formatting
      context _toplevel.update(s => {
        if s == none {
          file
        } else {
          s
        }
      })
      let cond() = _toplevel.final() == file
      project.with(..args, title: context meta.summary.find(x => x.at(0) == _toplevel.final()).at(1), cond: cond)([
        #show ref: it => context if _toplevel.final() == file {
          xref(it)
        }
        #context _xref-included.final().pairs().map(((key, value)) => context if value and cond() {
          xref-include(key)
        }).join()
        #body
      ])
    }
  } else {
    body => body
  }
}
