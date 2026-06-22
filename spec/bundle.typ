#import "/meta.typ": meta, common-formatting

#context assert(target() == "bundle", message: "Please compile this file only with `--format bundle`")

#set document(author: meta.authors, title: meta.title)
#set heading(numbering: "1.1")
#show: common-formatting

// HTML-specific stuff
// TODO: improve
#show align: it => it.body
#show grid: it => table(columns: it.columns, gutter: it.column-gutter, ..it.children.map(c => c.body))
#show math.frac.where(style: "skewed"): it => math.frac(it.num, it.denom, style: "horizontal")
#show math.equation: it => {
  show raw: r => r.text
  it
}

// TODO: navbar
#let nav = []

#document("/index.html", include "front.typ")

#for (partname, part) in meta.summary {
  for (name, title, ref) in part {
    document("/" + name + ".html")[
      #heading(level: 1, title)#ref
      #set heading(offset: 1)
      #include name + ".typ"
    ]
  }
}


// Waiting for something like https://github.com/typst/typst/issues/8309
// #document("/spec.pdf", include "spec.typ")<spec-pdf>
// #document("/spec.html", include "spec.typ")<spec-html>
