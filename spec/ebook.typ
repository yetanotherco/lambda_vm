#import "@preview/shiroa:0.3.1": *
#import "/book.typ": style

#import "/templates/ebook.typ"

#show: ebook.project.with(title: "typst-book", spec: "book.typ")
#style.update((
  foreground: black,
))

// set a resolver for inclusion
#ebook.resolve-inclusion(it => include it)
