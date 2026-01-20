#import "@preview/shiroa:0.3.1": *

#show: book

#book-meta(
  title: "Lambda VM specification",
  summary: [
    #chapter("variables.typ")[Variables]
    #chapter("is_bit.typ")[IS_BIT template]
    #chapter("add.typ")[ADD template]
    #chapter("decode.typ")[DECODE chip]
    #chapter("cpu.typ")[CPU chip]
    #chapter("shift.typ")[SHIFT chip]
    #chapter("branch.typ")[BRANCH]
    #chapter("lt.typ")[LT]
    #chapter("mul.typ")[MUL chip]
    #chapter("bitwise.typ")[BITWISE]
  ]
)

// re-export page template
#import "/templates/page.typ": project
#let book-page = project

#let todo(background: white, foreground: black, name: none, body) = block(fill: background, outset: 0.5em, radius: 20%, stroke: black)[
  #set text(fill: foreground)
  *TODO #if name != none { [(#name)] }*: #body
]
#let rj = todo.with(background: teal, name: "Robin")
#let et = todo.with(background: rgb("d4aa3a"), name: "Erik")
