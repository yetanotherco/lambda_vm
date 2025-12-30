
#import "@preview/shiroa:0.3.1": *

#show: book

#book-meta(
  title: "Lambda VM specification",
  summary: [
    #chapter("variables.typ")[Variables]
    #chapter("mul.typ")[MUL chip]
  ]
)

// re-export page template
#import "/templates/page.typ": project
#let book-page = project
