#import "/meta.typ": meta, common-formatting, highlights

#context assert(target() == "bundle", message: "Please compile this file only with `--format bundle`")

#set document(author: meta.authors, title: meta.title)
#set heading(numbering: "1.1")
#show: common-formatting

// Add an HTML attr to an element if not present. Helps guard against show rule recursion.
// `value` can also be a function depending on the element
#let add-attr(key, value) = it => {
  if key in it.attrs {
    it
  } else {
    let v = if type(value) == function { value(it) } else { value }
    html.elem(it.tag, attrs: it.attrs + ((key):v), it.body)
  }
}

// HTML-specific stuff
// TODO: improve
#show pagebreak: none
#show align: it => it.body
#show grid: it => table(columns: it.columns, gutter: it.column-gutter, ..it.children.map(c => c.body))
#show math.frac.where(style: "skewed"): it => math.frac(it.num, it.denom, style: "horizontal")
#show math.equation: it => {
  show raw: r => r.text
  it
}
#show selector.or(..highlights.keys().map(k => figure.where(kind: k))): it => {
  show html.elem.where(tag: "figure"): add-attr("data-kind", "highlight")
  show html.elem.where(tag: "figcaption"): add-attr("style", "background-color:" + highlights.at(it.kind).at(1).to-hex() + ";color:" + highlights.at(it.kind).at(2).to-hex() + ";")
  it
}
#show figure.where(kind: "thmenv"): fig => {
  show html.elem.where(tag: "figure"): add-attr("data-kind", lower(repr(fig.supplement).slice(1, -1)))
  show pad: it => it.body
  show h: none
  show figure.caption: none
  show "∎": html.elem("mrow", attrs: ("class": "qed"), "")
  fig
}
// TODO: todo callouts (rj/et/cdsg)
// TODO: table divider lines (vline/hline)
// TODO(a11y): replace table.header calls with custom functions to indicate "scope" (col/row/rowgroup) so that we can export that to the html th

#let nav(chapter) = {
  let content = meta.summary.map(((title, chapters)) => {
    strong(title);
    list(..chapters.map(((cname, ctitle, cref)) => {
      if cname == chapter {
        html.a(ctitle, href: "#", class: "current", aria-current: "page")
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
  let rellink(label, title, rel) = {
    show html.elem.where(tag: "a"): add-attr("rel", rel)
    link(label, title)
  }

  let flat = meta.summary.map(((_, chapters)) => chapters).sum(default: ())
  let index = flat.position(c => c.at(0) == chapter)
  if index == none {
    index = -1
  }

  html.nav(class: "prev-next",
    html.div(class: "prev",
      if index == 0 {
        rellink(label("doc:index"), meta.title, "prev")
      } else if index > 0 {
        let (name, title, _) = flat.at(index - 1)
        rellink(label("doc:" + name), title, "prev")
      }
    )
    +
    html.div(class: "next",
      if index < flat.len() - 1 {
        let (name, title, _) = flat.at(index + 1)
        rellink(label("doc:" + name), title, "next")
      }
    )
  )
}

#let chapter(filename, ctitle, mainbody) = [
  #let (doctitle, vistitle) = if ctitle == meta.title {
    (ctitle, ctitle)
  } else {
    (ctitle + " | " + meta.title, meta.title + html.span(class: "subheader", ctitle))
  }

  #document("/" + filename + ".html", title: doctitle, {
      html.link(href: "/style.css", rel: "stylesheet")
      html.link(href: "/fonts.css", rel: "stylesheet")
      html.link(href: "/sidenotes.css", rel: "stylesheet")
      html.script(src: "/sidenotes.js", defer: true)
      html.header(title(link(<doc:index>, vistitle)))
      html.main(mainbody)
      nav(filename)
      prev_next(filename)
  })#label("doc:"+filename)
]

#asset("/style.css", read("style.css"))
#asset("/fonts.css", read("fonts.css"))
#asset("/sidenotes.css", read("sidenotes.css"))
#asset("/sidenotes.js", read("sidenotes.js"))

// Bundled fonts
#for f in (
  read("fonts.css")
    .matches(regex("url\\(\"([^\"]+)\"\\)"))
    .map(m => m.captures.first())
) {
  asset("/" + f, read(f, encoding: none))
}

#chapter("index", meta.title, include "front.typ")
#for (partname, part) in meta.summary {
  for (name, title, ref) in part {
    chapter(name, title, [
      #heading(level: 1, title)#ref
      #set heading(offset: 1)
      #include name + ".typ"
    ])
  }
}


// Waiting for something like https://github.com/typst/typst/issues/8309
// #document("/spec.pdf", include "spec.typ")<spec-pdf>
// #document("/spec.html", include "spec.typ")<spec-html>
