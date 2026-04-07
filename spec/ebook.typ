#import "/book.typ": meta, common-formatting

#set document(author: meta.authors, title: meta.title)

#align(center, title(meta.title))
#align(center, text(style: "italic", fill: luma(40%))[Version #meta.version])
#align(center, meta.authors.join(", "))
#pagebreak(weak: true)
#outline()

#show: common-formatting
#show heading: set heading(numbering: "1.1")
#show raw.where(block: true): set block(fill: luma(230))

#meta.summary.map(((ch_title, sections)) => (
  [
   #pagebreak(weak: true)
   #heading(supplement: [Chapter], level: 1, ch_title)
  ]
  +
  sections.map(((sec, sec_title, ref)) => [
  #pagebreak(weak: true)
    #heading(supplement: [Section], level: 2, sec_title)#ref
    #set heading(offset: 2)
    #include sec
]).join()
)).join()
