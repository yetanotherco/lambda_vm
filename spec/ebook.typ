#import "/book.typ": style, meta

#set document(author: meta.authors, title: meta.title)

#style.update((
  foreground: black,
))

#set footnote(numbering: "[1]")

#align(center, title(meta.title))
#pagebreak(weak: true)
#outline()

#show heading: set heading(numbering: "1.1")

#meta.summary.map(((ch, title, ref)) => [
  #pagebreak(weak: true)
  #heading(supplement: [Chapter], level: 1, title)#ref
  #set heading(offset: 1)
  #include ch
]).join()
