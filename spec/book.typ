
#import "@preview/shiroa:0.3.1": *

#show: book

#book-meta(
  title: "Lambda VM specification",
  summary: [
    #prefix-chapter("sample_page.typ")[Sample page]
  ]
)

// re-export page template
#import "/templates/page.typ": project
#let book-page = project
