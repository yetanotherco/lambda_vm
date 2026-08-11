#import "@preview/shiroa:0.3.1": *
#import "/templates/page.typ": project
#import "@preview/equate:0.3.2": equate

#show: book

#let meta = (
  title: "Lambda VM specification",
  authors: ("3MI Labs", "Aligned"),
  version: "0.2",
  summary: (
    ("PROOF SYSTEM", (
        ("logup.typ", [`LogUp` argument], <logup>),
        ("memory.typ", [Memory argument], <memory>),
        ("streaming.typ", [Streaming prover], <streaming>),
    )),
    ("OVERVIEW", (
        ("variables.typ", [Variables], <vars>),
        ("signatures.typ", [Signatures], <signatures>),
    )),
    ("TEMPLATES", (
      ("is_bit.typ", [`IS_BIT` template], <isbit>),
      ("is_byte.typ", [`IS_BYTE` template], <isbyte>),
      ("sign.typ", [`SIGN` template], <sign>),
      ("add.typ", [`ADD`/`SUB` template], <add>),
      ("neg.typ", [`NEG` template], <neg>),
      ("reg.typ", [`REG`/`REGW` template], <reg>),
    )),
    ("CPU", (
      ("decode.typ", [`DECODE` table], <decode>),
      ("cpu.typ", [`CPU` chip], <cpu>),
      ("cpu32.typ", [`CPU32` chip], <cpu32>),
    )),
    ("ALU", (
      ("shift.typ", [`SHIFT` chip], <shift>),
      ("branch.typ", [`BRANCH` chip], <branch>),
      ("lt.typ", [`LT` chip], <lt>),
      ("eq.typ", [`EQ` chip], <eq>),
      ("mul.typ", [`MUL` chip], <mul>),
      ("dvrm.typ", [`DVRM` chip], <dvrm>),
      ("bitwise.typ", [`BITWISE` chips], <bitwise>),
      ("bytewise.typ", [`BYTEWISE` chip], <bytewise>)
    )),
    ("MEMORY", (
      ("memw.typ", [`MEMW` chip], <memw>),
      ("load.typ", [`LOAD` chip], <load>),
      ("store.typ", [`STORE` chip], <store>),
    )),
    ("ECALLS", (
      ("about_ecalls.typ", [About `ECALL`], <ecall>),
      ("halt.typ", [`HALT` chip], <halt>),
      ("commit.typ", [`COMMIT` chip], <commit>),
      ("sha256.typ", [`SHA256` accelerator], <sha256>),
      ("keccak.typ", [`KECCAK` accelerator], <keccak>),
      ("dma.typ", [`DMA` accelerator], <dma>),
      ("ecsm.typ", [`ECSM` accelerator], <ecsm>),
      ("fext.typ", [Extension field accelerator], <fext>),
    )),
    ("MATHEMATICS", (
      ("limbs_and_carries.typ", [On limb decomposition and carries], <limbs>),
    ))
  )
)
#let meta_sections = meta.summary.map(m => m.at(1)).sum()
#book-meta(
  title: meta.title,
  authors: meta.authors,
  summary: prefix-chapter("front.typ", meta.title) 
    + meta.summary.map(
      ((title, sections)) => {
        heading(depth: 1, title) + sections.map(((ch, title, _ref)) => chapter(ch, title)).join()
      }
    ).join()
)

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


#let todo(background: white, foreground: black, name: none, body) = block(fill: background, outset: 0.4em, radius: 20%, stroke: black)[
  #set text(fill: foreground)
  *TODO #if name != none { [(#name)] }*: #body
]
#let rj = todo.with(background: teal, name: "Robin")
#let et = todo.with(background: rgb("d4aa3a"), name: "Erik")
#let cdsg = todo.with(background: olive, name: "Cyprien")


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
  show ref: none
  context {
    place(hide(box(width: auto, height: 0%, strip-all(include "/" + f))))
  }
}

// Generate a cross-link for references to other chapters.
// Leaves the ref untouched if it can't be resolved or points to the current chapter.
#let xref(rf) = {
  assert(is-shiroa, message: "xref should only be used when compiling for shiroa")
  let lbl = rf.target
  let found = meta_sections.find(((_, _, tag)) => str(lbl).starts-with(str(tag)))
  context if found != none and found.at(0) != _toplevel.final() {
    let (ch, title, ref) = found
    if ref == lbl {
      cross-link("/" + ch, [Chapter #(meta_sections.position(x => x == found) + 1)])
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
        [#supplement #numbering(fig.numbering, ..counter.at(lbl))]
      }
      cross-link("/" + ch, reference: shiroa-label, link-content)
    }
  } else {
    rf
  }
}

#let book-page(file, ..args) = {
  if not file.ends-with(".typ") {
    file = lower(file) + ".typ"
  }

  assert(meta_sections.find(s => s.at(0) == file) != none, message: "Couldn't resolve typst source file " + file)

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
      show ref: it => context if cond() { xref(it) }
      let title = context {
        // Strip raw, because shiroa already makes the title raw
        show raw: it => it.text
        meta_sections.find(x => x.at(0) == _toplevel.final()).at(1)
      }
      project.with(..args, title: title, description: plain-text(meta_sections.find(x => x.at(0) == file).at(1)), cond: cond)([
        #context _xref-included.final().pairs().map(((key, value)) => context if value and cond() {
          xref-include(key)
        }).join()
        #metadata(json("interaction_count.json").sum(default: (:)))<interaction_count>

        #let chapter-index = meta_sections.position(x => x.at(0) == file) + 1
        #set heading(numbering: (..args) => [#chapter-index.#numbering("1.1", ..args)])
        #counter(heading).update(0)

        #body
      ])
    }
  } else {
    body => body
  }
}
