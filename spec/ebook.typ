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

#meta.summary.map(((ch, title, ref)) => [
  #pagebreak(weak: true)
  #heading(supplement: [Chapter], level: 1, title)#ref
  #set heading(offset: 1)
  #include ch
]).join()
