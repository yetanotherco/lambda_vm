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
// TODO: aside, todo callouts (rj/et/cdsg), the alternating table colors of decode

#let nav(chapter) = {
  let content = meta.summary.map(((title, chapters)) => {
    strong(title);
    list(..chapters.map(((cname, ctitle, cref)) => {
      if cname == chapter {
        html.a(ctitle, class: "current", tabindex: -1)
      } else {
        link(label("doc:" + str(cname)), ctitle)
      }
    }))
  }).join()

  html.nav({
    html.div(class: "desktop-nav", content)
    html.details(class: "mobile-nav", html.summary("Navigation") + content)
  })
}

#let prev_next(chapter) = {
  let flat = meta.summary.map(((_, chapters)) => chapters).sum(default: ())
  let index = flat.position(c => c.at(0) == chapter)
  if index == none {
    index = -1
  }

  html.nav(class: "prev-next",
    html.div(class: "prev",
      if index == 0 {
        link(label("doc:index"), meta.title)
      } else if index > 0 {
        let (name, title, _) = flat.at(index - 1)
        link(label("doc:" + name), title)
      }
    )
    +
    html.div(class: "next",
      if index < flat.len() - 1 {
        let (name, title, _) = flat.at(index + 1)
        link(label("doc:" + name), title)
      }
    )
  )
}

#let chapter(filename, title, mainbody) = [
  #document("/" + filename + ".html", title: title, {
      html.link(href: "/style.css", rel: "stylesheet")
      heading(numbering: none, link(<doc:index>, meta.title))
      nav(filename)
      html.main(mainbody)
      prev_next(filename)
  })#label("doc:"+filename)
]

#asset("/style.css", read("style.css"))
// TODO? sidenotes
#chapter("index", meta.title, include "front.typ")

#for (partname, part) in meta.summary {
  for (name, title, ref) in part {
    chapter(name, title + " | " + meta.title, [
      #heading(level: 1, title)#ref
      #set heading(offset: 1)
      #include name + ".typ"
    ])
  }
}


// Waiting for something like https://github.com/typst/typst/issues/8309
// #document("/spec.pdf", include "spec.typ")<spec-pdf>
// #document("/spec.html", include "spec.typ")<spec-html>
