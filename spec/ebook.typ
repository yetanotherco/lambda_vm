#import "@preview/shiroa:0.3.1": *
#import "/book.typ": style, meta

#set document(author: meta.authors, title: meta.title)

#show heading: set heading(numbering: "1.1")

#style.update((
  foreground: black,
))

#align(center, title(meta.title))
// #outline()

#meta.summary.map(((ch, title, ref)) => [
  #heading(supplement: [Chapter], level: 1, title)#ref
  #include ch
]).join()
