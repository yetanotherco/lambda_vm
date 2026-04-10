#import "/book.typ": book-page
#import "/src.typ": load_signatures, load_config

#show: book-page("signatures.typ")

#let config = load_config()
#let signatures = load_signatures(config)

// Render a signature
#let render_signature(sig) = {
  let (lb, rb) = if sig.kind == "interaction" {
    (`[`, `]`)
  } else if sig.kind == "template" {
    (`<`, `>`)
  }

  let cond = sig.at("cond", default: none)
  let cond_str = if cond != none {
    raw(cond) + ` => `
  } else {``}

  let input_str = sig.input.map(elt => {
    if type(elt) == array {
      raw(elt.at(0)) + `[` + raw(str(elt.at(1))) + `]`
    } else {
      raw(elt)
    }
  }).join(`, `)

  let output = sig.at("output", default: none)
  let output_str = if output != none {
    if type(output) == array {
      raw(output.at(0)) + `[` + raw(str(output.at(1))) + `]`
    } else {
      raw(output)
    } + `; `
  } else {``}

  return [#cond_str#raw(sig.tag)#lb#output_str#input_str#rb]
}

// Compute the bus size of an interaction
#let interaction_bus_size(sig) = {
  let vars = sig.input + if "output" in sig { (sig.output, )} else {()}

  return vars.map(v => {
    let (label, factor) = if type(v) == array {
      (v.at(0), v.at(1))
    } else {
      (v, 1)
    }
    config.variables.types.filter(type => type.label == label).first().subtypes.len() * factor
  })
  .sum()
}

#let interactions = signatures.signatures.filter(s => s.kind == "interaction")
The following lists signatures of the #interactions.len() interactions in this VM.
#figure(table(
    columns: (1fr, auto),
    inset: 7pt,
    align: (top+left, center),
    stroke: none,
    table.header([*Signature*], [*Bus size*]),
    table.hline(stroke: 1pt),
    table.vline(stroke: 1pt, x: 1),
    ..for sig in interactions {
      ([#render_signature(sig)], [#interaction_bus_size(sig)])
    },
))

#let templates = signatures.signatures.filter(s => s.kind == "template")
Below, we list the signatures of the #templates.len() templates in this VM.
#figure(table(
    columns: 1fr,
    inset: 7pt,
    align: (top+left, center),
    stroke: none,
    table.header([*Signature*]),
    table.hline(stroke: 1pt),
    ..for sig in templates {
      ([#render_signature(sig)], )
    },
))
