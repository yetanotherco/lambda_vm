#import "/meta.typ": meta, common-formatting

#set document(author: meta.authors, title: meta.title)

#align(center, title(meta.title))
#align(center, text(style: "italic", fill: luma(40%))[Version #meta.version])
#align(center, meta.authors.join(", "))
#pagebreak(weak: true)

// outline
#show outline.entry.where(level: 1): set outline.entry(fill: line(length: 100%, stroke: stroke(dash: "solid")))
#show outline.entry.where(level: 1): it => {
  v(15pt, weak: true)
  strong(it)
  v(5pt, weak: true)
}
#show outline.entry.where(level: 2): it => {
  v(10pt, weak: true)
  it
}
#outline(depth: 3)

// chapter pages
#show heading.where(level: 1): it => align(center + horizon)[#underline(it, offset: 10pt, extent: 5pt)]

#show: common-formatting
#show heading: set heading(numbering: (..args) => {
    let args = args.pos()
    let skip_first = args.slice(calc.min(args.len(), 1))
    numbering("1.1", ..skip_first)
})
#show raw.where(block: true): set block(fill: luma(230))

#for (ch_title, sections) in meta.summary {
  pagebreak(weak: true)
  heading(supplement: [Chapter], level: 1, ch_title, numbering: none)
  
  for (sec, sec_title, ref) in sections {
    pagebreak(weak: true)
    [#heading(level: 2, supplement: [Section], sec_title)#ref]
    set heading(offset: 2)
    include sec + ".typ"
  }
}
